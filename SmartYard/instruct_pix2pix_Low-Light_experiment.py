import os
import shutil
import random
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리

# --- 1. 모델 및 스케줄러 설정 ---
print("InstructPix2Pix 모델을 로드합니다. 잠시만 기다려주세요...")
try:
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    print("모델 로드가 완료되었습니다.")
except Exception as e:
    print(f"모델 로드 중 오류가 발생했습니다: {e}")
    print("CUDA 설정 또는 인터넷 연결을 확인해주세요.")
    exit()

# --- 2. 저조도 변환을 위한 프롬프트 정의 ---
# 'l'은 light, 'h'는 heavy를 의미합니다. 파일명 접미사로 사용됩니다.
low_light_prompts = {
    "l": "Make the image look like it was taken at night with a low-light source. Make it slightly dark, but maintain a neutral color balance.",
    "h": "Make the image look like it was taken at night with very little light source. Make it very dark, but maintain a neutral color balance."
}

# --- 3. 이미지 로드 헬퍼 함수 ---
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

# --- 4. 데이터셋 증강을 위한 메인 함수 (수정된 버전) ---
def augment_dataset_for_yolo(base_dir, augmented_only_root_dir):
    """
    지정된 디렉토리와 그 하위 모든 디렉토리의 이미지에 대해 저조도 증강을 수행하고,
    결과를 (1)기존 경로와 (2)별도 경로에 모두 저장합니다. (os.walk 사용)
    """
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"경고: '{images_dir}' 또는 '{labels_dir}'를 찾을 수 없습니다. 이 디렉토리를 건너뜁니다.")
        return

    dataset_type = os.path.basename(base_dir)
    augmented_images_dir = os.path.join(augmented_only_root_dir, dataset_type, "images")
    augmented_labels_dir = os.path.join(augmented_only_root_dir, dataset_type, "labels")

    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_labels_dir, exist_ok=True)
    print(f"증강된 파일은 '{os.path.join(augmented_only_root_dir, dataset_type)}' 폴더에 별도로 저장됩니다.")

    image_paths = []
    for dirpath, _, filenames in os.walk(images_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "_low_light_" not in filename:
                    image_paths.append(os.path.join(dirpath, filename))

    print(f"\n{'='*60}")
    print(f"'{base_dir}' 디렉토리 처리 시작... 총 {len(image_paths)}개의 원본 이미지 발견")
    print(f"{'='*60}")

    for original_image_path in tqdm(image_paths, desc=f"Processing {os.path.basename(base_dir)}"):
        filename = os.path.basename(original_image_path)
        name_part, ext_part = os.path.splitext(filename)

        relative_path = os.path.relpath(original_image_path, images_dir)
        relative_label_path = os.path.splitext(relative_path)[0] + ".txt"
        original_label_path = os.path.join(labels_dir, relative_label_path)

        if not os.path.exists(original_label_path):
            tqdm.write(f"경고: '{filename}'에 대한 라벨 파일 '{original_label_path}'가 없습니다. 건너뜁니다.")
            continue

        base_image = load_image(original_image_path)
        if base_image is None:
            tqdm.write(f"경고: '{filename}' 이미지를 로드할 수 없어 건너뜁니다.")
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

            # --- ✨ 수정된 부분 시작 ---
            # 이미 생성된 파일이 두 경로에 모두 존재하면 건너뜁니다.
            if os.path.exists(new_image_path_orig) and os.path.exists(new_image_path_aug_only):
                tqdm.write(f"파일이 이미 존재하여 건너뜁니다: {new_image_filename}")
                continue
            # --- ✨ 수정된 부분 끝 ---

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
                tqdm.write(f"오류: '{filename}' 이미지 생성 중 문제 발생 (강도: {intensity}). 오류: {e}")

# --- 5. Run the augmentation process ---
if __name__ == "__main__":
    yolo_base_path = "/SSD4/psleon/YOLOv11/data"
    
    augmented_only_base_path = os.path.join(yolo_base_path, "Augmented_Only_Data")
    
    training_path = os.path.join(yolo_base_path, "Training")
    validation_path = os.path.join(yolo_base_path, "Validation")

    augment_dataset_for_yolo(training_path, augmented_only_base_path)
    augment_dataset_for_yolo(validation_path, augmented_only_base_path)

    print(f"\n{'='*60}")
    print("모든 데이터 증강 작업이 완료되었습니다!")
    print(f"증강된 파일은 '{augmented_only_base_path}' 폴더에서 별도로 확인 가능")
    print(f"{'='*60}")