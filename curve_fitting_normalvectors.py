import numpy as np
import paddle
from paddleseg.models import UNet
from PIL import Image
import os
import glob
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sys

# get argument from user
arg = sys.argv[1]
input_folder = f'data/png/{arg}'
output_folder = f'data/output/{arg}'

# Pre-trained 모델 불러오기 및 설정
model = UNet(num_classes=2)  # 클래스 수에 맞게 설정
model_path = 'model.pdparams'  # 다운로드한 Pre-trained 모델(unet)의 경로
model.set_dict(paddle.load(model_path))
model.eval()

# 3차 다항식 함수 정의
def third_order_curve(x, a0, a1, a2, a3):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3

# 법선 벡터 기울기 계산
def calculate_normal_slope(x, y):
    dy_dx = np.gradient(y, x)
    normal_slope = -1 / dy_dx
    return normal_slope
    
# 세그멘테이션 수행 및 결과 저장
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        input_data = paddle.to_tensor(image)

        with paddle.no_grad():
            output = model(input_data)

        output = [paddle.to_tensor(output_item) for output_item in output]
        argmax_output = paddle.argmax(output[0], axis=1)
        segmentation_result = argmax_output.numpy()[0]

        colors = [
            [0, 0, 0],   # 배경: 검은색
            [255, 0, 0]  # 혈관: 빨간색
        ]

        colored_segmentation = np.zeros((segmentation_result.shape[0], segmentation_result.shape[1], 3), dtype=np.uint8)
        for class_idx, color in enumerate(colors):
            mask = segmentation_result == class_idx
            for c in range(3):
                colored_segmentation[mask, c] = color[c]

        # 혈관(빨간색)으로 인식된 픽셀 식별
        vessel_pixels = np.where(segmentation_result == 1)  # 1은 혈관 클래스에 해당하는 인덱스입니다
        
        # 혈관 픽셀 좌표 추출
        vessel_pixels_x = vessel_pixels[1]
        vessel_pixels_y = vessel_pixels[0]

        # 3차 다항식으로 curve fitting
        params, covariance = curve_fit(third_order_curve, vessel_pixels_x, vessel_pixels_y,
                                p0=(0, 0, 0, 0))  # 초기 추정값은 0으로 설정하거나 필요에 따라 수정
        a_fit, a1_fit, a2_fit, a3_fit = params

        # Fitting된 곡선 좌표 계산
        fit_curve_x = np.linspace(min(vessel_pixels_x), max(vessel_pixels_x), 100)
        fit_curve_y = third_order_curve(fit_curve_x, a_fit, a1_fit, a2_fit, a3_fit)
       
        # 시각화 및 저장
        plt.figure()
        plt.imshow(segmentation_result, cmap='gray')
        plt.plot(fit_curve_x, fit_curve_y, color=(1, 0, 0, 0.5), linewidth=2)
        plt.axis('off')
        seg_image_path =os.path.join(output_folder, filename.replace('.png', '_cropped_curve.png'))
        plt.savefig(seg_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Segmentation result with curve saved")

        normal_slopes = calculate_normal_slope(fit_curve_x, fit_curve_y)

        # 결과 저장
        result_file_path = os.path.join(output_folder, filename.replace('.png', '_normal_vector.txt'))
        with open(result_file_path, 'w') as result_file:
            for x_point, y_point, slope in zip(fit_curve_x, fit_curve_y, normal_slopes):
                result_file.write(f"{x_point}, {y_point} : {slope}\n")
