import os
import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rc('font', family='NanumGothic')
mpl.rcParams['axes.unicode_minus'] = False

from ultralytics import YOLO

def pre_flight_path_check(data_yaml_path):
    """
    학습 시작 전, dataset.yaml 파일에 명시된 경로가 유효한지 검사합니다.
    """
    print("\n[사전 경로 검증 시작]")
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # train, val 경로가 있는지 확인
        if 'train' not in data or 'val' not in data:
            print(f"❌ [경로 검증 실패] '{data_yaml_path}' 파일에 'train' 또는 'val' 키가 없습니다.")
            return False

        # project_root는 train.py가 실행되는 위치입니다.
        project_root = os.getcwd() 
        
        # 경로 조합 및 검증
        paths_to_check = {
            'Train': os.path.join(project_root, 'data', data['train']),
            'Validation': os.path.join(project_root, 'data', data['val'])
        }

        all_paths_found = True
        for name, path in paths_to_check.items():
            print(f"  - {name} 경로 확인 중: {path}")
            if not os.path.isdir(path):
                print(f"  ❌ [경로 검증 실패] '{name}' 데이터 폴더를 찾을 수 없습니다.")
                print(f"     예상 경로: {path}")
                print(f"     터미널에서 'ls -ld {path}' 명령어로 실제 경로가 맞는지 확인해보세요.")
                all_paths_found = False
        
        if all_paths_found:
            print("✅ [경로 검증 성공] 모든 데이터 경로를 성공적으로 찾았습니다.")
            return True
        else:
            return False

    except FileNotFoundError:
        print(f"❌ [오류] '{data_yaml_path}' 파일을 찾을 수 없습니다. 파일 위치를 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ [오류] '{data_yaml_path}' 파일을 읽는 중 문제가 발생했습니다: {e}")
        return False

data_yaml_path = 'dataset.yaml'

if not pre_flight_path_check(data_yaml_path):
    print("\n데이터셋 경로 문제로 학습을 시작할 수 없습니다. 위의 오류 메시지를 확인하고 수정해주세요.")


print("="*20)
print("YOLO Custom Training Starts")
print("="*20)

data_yaml_path = 'dataset.yaml'
model_name = 'yolo11n.pt'

model = YOLO(model_name)
# 학습 파라미터
epochs = 100      # 전체 데이터셋을 몇 번 반복 학습할지 결정
img_size = 640    # 학습에 사용될 이미지 크기
batch_size = 16   # 한 번에 처리할 이미지 수 (GPU VRAM에 따라 조절)
run_name = 'outputs' # 학습 결과가 저장될 폴더 이름

print(f"- 데이터셋: {os.path.abspath(data_yaml_path)}")
print(f"- 시작 모델: {model_name}")
print(f"- 에포크: {epochs}")
print(f"- 이미지 크기: {img_size}")
print(f"- 배치 사이즈: {batch_size}")

results = model.train(
    data=data_yaml_path,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    name=run_name,
    device=[5, 6],
    exist_ok=True)

print("학습이 성공적으로 완료되었습니다!")
# 가장 성능이 좋았던 모델(best.pt)의 경로를 출력
# 이 경로는 나중에 inference.py에서 사용됩니다.
best_model_path = results.save_dir / 'weights/best.pt'
print(f"가장 성능이 좋은 모델이 다음 경로에 저장되었습니다:\n{best_model_path}")