import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

import os

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def load_image(image_path):
    image = PIL.Image.open(image_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# --- 설정: 날씨 조건 및 프롬프트 정의 ---
weather_conditions = {
    "haze": {
        "light": "subtly apply a very thin layer of haze, slightly softening the background",
        "heavy": "the scene is enveloped in dense, thick fog, significantly reducing visibility"
    },
    "rain": {
        "light": "make the image look as if it's in a light drizzle, with slightly damp surfaces",
        "heavy": "add a heavy downpour of rain, with large puddles and streaks of rain"
    },
    "snow": {
        "light": "a few snowflakes are gently falling, with a very light dusting of snow on some surfaces",
        "heavy": "a heavy snowstorm is happening, covering everything in a thick blanket of snow"
    }
}
num_images_per_condition = 10


image_path = "3.jpg"

base_image = load_image(image_path)

if base_image:
    # --- 메인 루프: 각 날씨의 강도별로 이미지 생성 (확장된 부분) ---
    total_images_generated = 0
    # 바깥쪽 루프: 날씨 조건 순회 (haze -> rain -> snow)
    for condition, intensities in weather_conditions.items():
        # 안쪽 루프: 강도 순회 (light -> heavy)
        for intensity, prompt in intensities.items():
            print(f"\n{'='*50}")
            print(f"처리 중: [날씨: {condition}] / [강도: {intensity}]")
            print(f"{'='*50}")

            # 1. 결과물을 저장할 폴더 생성 (예: 'outputs/haze/light')
            output_dir = os.path.join("outputs", condition, intensity)
            os.makedirs(output_dir, exist_ok=True)
            print(f"결과물 저장 폴더: '{output_dir}'")

            # 2. 파이프라인 호출하여 이미지 10개 생성
            print(f"사용될 프롬프트: '{prompt}'")
            print(f"이미지 {num_images_per_condition}개 생성을 시작합니다...")
            
            images = pipe(
                prompt,
                image=base_image,
                num_images_per_prompt=num_images_per_condition,
                num_inference_steps=15,
                image_guidance_scale=1.5
            ).images
            
            print("이미지 생성이 완료되었습니다. 파일 저장을 시작합니다.")

            # 3. 생성된 이미지들을 해당 폴더에 저장
            for i, img in enumerate(images):
                # 파일 이름을 'haze_light_1.png' 와 같이 설정
                save_path = os.path.join(output_dir, f"{condition}_{intensity}_{i+1}.png")
                img.save(save_path)
            
            total_images_generated += len(images)
            print(f"'{condition} - {intensity}' 조건의 이미지 {num_images_per_condition}개를 모두 저장했습니다.")

    print(f"\n{'='*50}")
    print(f"모든 작업이 완료되었습니다. 총 {total_images_generated}개의 이미지가 생성되었습니다.")
    print(f"{'='*50}")