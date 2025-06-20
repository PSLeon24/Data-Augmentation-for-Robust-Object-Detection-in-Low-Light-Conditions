# inference.py

import os
from ultralytics import YOLO
from glob import glob

def inference_with_custom_model():
    """
    커스텀 학습된 YOLO 모델로 새로운 이미지에 대해 추론을 수행합니다.
    """
    print("="*60)
    print("커스텀 모델로 추론을 시작합니다.")
    print("="*60)

    # --- 1. 설정값 정의 ---
    model_path = 'runs/detect/outputs/weights/best.pt'

    # 추론할 이미지가 있는 폴더 경로
    image_source_path = 'Sublabel/'
    
    # 추론 결과가 저장될 폴더 이름
    run_name = 'inference_results'

    # --- 2. 모델 로드 ---
    if not os.path.exists(model_path):
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_path}")
        return
        
    if not os.path.exists(image_source_path):
        print(f"❌ 오류: 추론할 이미지 소스를 찾을 수 없습니다: {image_source_path}")
        return

    # 커스텀 학습된 가중치로 모델을 로드
    model = YOLO(model_path)
    
    print("\n[추론 설정]")
    print(f"- 커스텀 모델: {os.path.abspath(model_path)}")
    print(f"- 추론 대상: {os.path.abspath(image_source_path)}")
    print("-" * 30)

    # --- 3. 추론 실행 ---
    results = model.predict(
        source=image_source_path,
        conf=0.5,       # confidence threshold: 50% 이상 확신하는 객체만 표시
        save=True,      # 바운딩 박스가 그려진 이미지 저장
        name=run_name,
        exist_ok=True
    )
    
    # 결과 객체에서 저장된 경로 정보 추출
    # predict()는 리스트를 반환하므로 첫 번째 요소의 save_dir를 사용
    saved_dir = ''
    if results and len(results) > 0:
        saved_dir = results[0].save_dir

    print("\n" + "="*60)
    print("추론이 성공적으로 완료되었습니다!")
    if saved_dir:
        print(f"결과가 다음 폴더에 저장되었습니다:\n{os.path.abspath(saved_dir)}")
    print("="*60)


if __name__ == '__main__':
    inference_with_custom_model()