from PIL import Image
import numpy as np
import os

def analyze_and_visualize_difference(file1, file2, diff_save_path='difference_map.png'):
    """
    두 이미지의 차이를 분석하고, Difference Map을 시각화하여 저장합니다.

    Args:
        file1 (str): 첫 번째 이미지 파일 경로
        file2 (str): 두 번째 이미지 파일 경로
        diff_save_path (str): 차이 맵을 저장할 경로
    """
    try:
        # 이미지 파일 열기
        img1 = Image.open(file1)
        img2 = Image.open(file2)

        # 이미지 크기와 모드가 동일한지 확인
        if img1.size != img2.size or img1.mode != img2.mode:
            print("❌ 이미지의 크기나 모드가 달라 비교할 수 없습니다.")
            return

        # 이미지를 NumPy 배열로 변환
        # 연산을 위해 부동소수점 형태로 변환합니다.
        img1_data = np.array(img1).astype(float)
        img2_data = np.array(img2).astype(float)

        # --- 1. 차이점 정량 분석 ---
        # 픽셀 값의 절대적인 차이 계산
        diff = np.abs(img1_data - img2_data)

        # 다른 픽셀 개수 계산
        total_pixels = img1.size[0] * img1.size[1]
        # 채널(RGB)을 모두 더했을 때 0보다 크면 다른 픽셀로 간주
        num_different_pixels = np.count_nonzero(np.sum(diff, axis=2) > 0)
        percentage_diff = (num_different_pixels / total_pixels) * 100

        # 평균 제곱 오차(MSE) 계산
        mse = np.mean(np.square(diff))

        print("--- 📊 이미지 차이 분석 결과 ---")
        if num_different_pixels == 0:
            print("✅ 두 이미지는 완전히 동일합니다.")
            return

        print(f"다른 픽셀 수: {num_different_pixels} / {total_pixels} 개")
        print(f"차이 비율: {percentage_diff:.4f} %")
        print(f"평균 제곱 오차 (MSE): {mse:.4f}")

        # --- 2. Difference Map 생성 및 저장 ---
        # 차이 배열을 0-255 범위의 8비트 정수로 변환하여 이미지로 만듭니다.
        # 차이가 클수록 더 밝게 표현됩니다.
        diff_map_data = diff.astype(np.uint8)
        diff_map_image = Image.fromarray(diff_map_data, mode=img1.mode)
        diff_map_image.save(diff_save_path)

        print(f"\n✅ Difference Map이 '{os.path.abspath(diff_save_path)}' 경로에 저장되었습니다.")
        print("   (검은색 = 동일한 픽셀, 밝은색 = 다른 픽셀)")


    except FileNotFoundError:
        print(f"❌ 오류: '{file1}' 또는 '{file2}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

# --- 코드 실행 부분 ---
if __name__ == "__main__":
    image1_path = '/SSD4/psleon/SmartYard/outputs/low-light/light/low-light_light_1.png'
    image2_path = '/SSD4/psleon/SmartYard/outputs/low-light/light/low-light_light_2.png'
    
    # 함수 호출
    analyze_and_visualize_difference(image1_path, image2_path)