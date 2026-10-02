import os
import shutil
import random
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tqdm import tqdm
import multiprocessing  # 멀티프로세싱 라이브러리 추가
import time # 시간 측정을 위해 추가

# --- 1. 저조도 변환을 위한 프롬프트 정의 ---
# 'l'은 light, 'h'는 heavy를 의미합니다. 파일명 접미사로 사용됩니다.
low_light_prompts = {
    "l": "Make the image look like it was taken at night with a low-light source. Make it slightly dark, but maintain a neutral color balance.",
    "h": "Make the image look like it was taken at night with very little light source. Make it very dark, but maintain a neutral color balance."
}

# --- 2. 이미지 로드 헬퍼 함수 ---
def load_image(image_path):
    """지정된 경로의 이미지를 로드하고 RGB로 변환합니다."""
    try:
        image = PIL.Image.open(image_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    except PIL.UnidentifiedImageError:
        print(f"오류: 이미지 파일을 식별할 수 없습니다: {image_path}")
        return None
    except Exception as e:
        print(f"이미지 로드 중 오류 발생 ({image_path}): {e}")
        return None

# --- 3. 각 GPU에서 실행될 작업 함수 ---
def process_images_on_gpu(worker_id, gpu_id, image_paths_chunk, base_dir, augmented_only_root_dir):
    """
    단일 GPU에서 할당된 이미지 목록을 처리하는 함수입니다.
    이 함수는 별도의 프로세스로 실행됩니다.
    """
    # **중요**: 이 프로세스가 지정된 GPU만 사용하도록 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 프로세스별로 필요한 라이브러리를 다시 import 할 수 있습니다.
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    print(f"[Worker {worker_id} on GPU {gpu_id}] 프로세스 시작. 모델을 로드합니다.")
    try:
        # 각 프로세스는 모델을 자체 GPU 메모리에 독립적으로 로드합니다.
        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        pipe.to("cuda") # "cuda"는 이제 지정된 GPU(예: gpu_id)를 가리킵니다.
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        print(f"[Worker {worker_id} on GPU {gpu_id}] 모델 로드 완료.")
    except Exception as e:
        print(f"[Worker {worker_id} on GPU {gpu_id}] 모델 로드 중 오류: {e}")
        return

    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    dataset_type = os.path.basename(base_dir)
    augmented_images_dir = os.path.join(augmented_only_root_dir, dataset_type, "images")
    augmented_labels_dir = os.path.join(augmented_only_root_dir, dataset_type, "labels")

    # tqdm의 position 인자를 사용하여 진행률 표시줄이 겹치지 않게 합니다.
    for original_image_path in tqdm(image_paths_chunk, desc=f"GPU {gpu_id} ({dataset_type})", position=worker_id):
        filename = os.path.basename(original_image_path)
        name_part, ext_part = os.path.splitext(filename)

        relative_path = os.path.relpath(original_image_path, images_dir)
        relative_label_path = os.path.splitext(relative_path)[0] + ".txt"
        original_label_path = os.path.join(labels_dir, relative_label_path)

        if not os.path.exists(original_label_path):
            continue

        base_image = load_image(original_image_path)
        if base_image is None:
            continue

        for intensity, prompt in low_light_prompts.items():
            suffix = f"_low_light_{intensity}"
            new_image_filename = f"{name_part}{suffix}{ext_part}"
            new_label_filename = f"{name_part}{suffix}.txt"
            
            relative_dir = os.path.dirname(relative_path)

            new_image_path_orig = os.path.join(images_dir, relative_dir, new_image_filename)
            new_label_path_orig = os.path.join(labels_dir, relative_dir, new_label_filename)
            new_image_path_aug_only = os.path.join(augmented_images_dir, relative_dir, new_image_filename)
            new_label_path_aug_only = os.path.join(augmented_labels_dir, relative_dir, new_label_filename)
            
            os.makedirs(os.path.dirname(new_image_path_orig), exist_ok=True)
            os.makedirs(os.path.dirname(new_label_path_orig), exist_ok=True)
            os.makedirs(os.path.dirname(new_image_path_aug_only), exist_ok=True)
            os.makedirs(os.path.dirname(new_label_path_aug_only), exist_ok=True)

            if os.path.exists(new_image_path_orig) and os.path.exists(new_image_path_aug_only):
                continue
            
            try:
                seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator("cuda").manual_seed(seed)
                generated_image = pipe(
                    prompt, image=base_image, num_images_per_prompt=1, num_inference_steps=15,
                    image_guidance_scale=1.5, guidance_scale=7.5, generator=generator,
                ).images[0]

                generated_image.save(new_image_path_orig)
                shutil.copy2(original_label_path, new_label_path_orig)
                
                generated_image.save(new_image_path_aug_only)
                shutil.copy2(original_label_path, new_label_path_aug_only)

            except Exception as e:
                tqdm.write(f"GPU {gpu_id} 오류: '{filename}' 생성 중 문제 발생. 오류: {e}")

# --- 4. 데이터셋 증강을 위한 메인 함수 (수정된 버전) ---
def augment_dataset_for_yolo_multi_gpu(base_dir, augmented_only_root_dir, gpu_ids):
    """
    지정된 디렉토리의 이미지를 여러 GPU를 사용하여 병렬로 증강합니다.
    """
    images_dir = os.path.join(base_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"경고: '{images_dir}'를 찾을 수 없습니다. 이 디렉토리를 건너뜁니다.")
        return

    # 증강 파일 저장 폴더 생성
    dataset_type = os.path.basename(base_dir)
    augmented_images_dir = os.path.join(augmented_only_root_dir, dataset_type, "images")
    augmented_labels_dir = os.path.join(augmented_only_root_dir, dataset_type, "labels")
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_labels_dir, exist_ok=True)

    # 처리할 모든 이미지 파일 경로 수집
    image_paths = []
    for dirpath, _, filenames in os.walk(images_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "_low_light_" not in filename:
                    image_paths.append(os.path.join(dirpath, filename))

    if not image_paths:
        print(f"'{base_dir}'에서 처리할 원본 이미지를 찾지 못했습니다.")
        return

    print(f"\n{'='*60}")
    print(f"'{base_dir}' 디렉토리 처리 시작... 총 {len(image_paths)}개의 원본 이미지 발견")
    print(f"사용할 GPU: {gpu_ids}")
    print(f"{'='*60}")

    # 이미지 목록을 GPU 개수만큼 분할
    num_gpus = len(gpu_ids)
    chunk_size = len(image_paths) // num_gpus
    if len(image_paths) % num_gpus != 0:
        chunk_size += 1
    
    image_path_chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        if i < len(image_path_chunks): # 작업할 이미지가 있는 경우에만 프로세스 생성
            process = multiprocessing.Process(
                target=process_images_on_gpu,
                args=(i, gpu_id, image_path_chunks[i], base_dir, augmented_only_root_dir)
            )
            processes.append(process)
            process.start()

    # 모든 프로세스가 끝날 때까지 대기
    for process in processes:
        process.join()


# --- 5. 메인 실행 블록 ---
if __name__ == "__main__":
    # 중요: 멀티프로세싱 시작 방식을 'spawn'으로 설정 (CUDA와 호환성을 위해)
    multiprocessing.set_start_method('spawn', force=True)

    yolo_base_path = "/SSD4/psleon/YOLOv11/data"
    augmented_only_base_path = os.path.join(yolo_base_path, "Augmented_Only_Data")
    
    training_path = os.path.join(yolo_base_path, "Training")
    validation_path = os.path.join(yolo_base_path, "Validation")

    # 사용할 GPU ID 목록
    target_gpu_ids = [0, 2, 3, 4]

    start_time = time.time()
    
    # Training 데이터셋 증강
    augment_dataset_for_yolo_multi_gpu(training_path, augmented_only_base_path, target_gpu_ids)
    
    # Validation 데이터셋 증강
    augment_dataset_for_yolo_multi_gpu(validation_path, augmented_only_base_path, target_gpu_ids)

    end_time = time.time()
    print(f"\n{'='*60}")
    print("모든 데이터 증강 작업이 완료되었습니다!")
    print(f"총 소요 시간: {end_time - start_time:.2f}초")
    print(f"증강된 파일은 '{augmented_only_base_path}' 폴더에서 별도로 확인 가능")
    print(f"{'='*60}")