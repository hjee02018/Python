import tkinter as tk
from tkinter import messagebox
import numpy as np
import paddle
from paddleseg.models import UNet
from PIL import Image
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sys
import math

arg = sys.argv[1]
dicom_folder = f'data/dicom/{arg}'
png_folder = f'data/png/{arg}'
output_folder = f'data/output/{arg}'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(png_folder, exist_ok=True)

# Pre-trained 모델 불러오기 및 설정
model = UNet(num_classes=2)  # 클래스 수에 맞게 설정
model_path = 'model.pdparams'  # 다운로드한 Pre-trained 모델(unet)의 경로
model.set_dict(paddle.load(model_path))
model.eval()

def third_order_curve(x, a0, a1, a2, a3):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3

# 미분 함수 정의
def derivative_third_order_curve(x, a1, a2, a3):
    return a1 + 2 * a2 * x + 3 * a3 * x**2

def get_end(x, y, slope):
    # 길이가 1인 선분의 끝점 계산
    length = 1
    x_end = x + (length / math.sqrt(1 + (slope**2)))
    y_end = y + (slope/math.sqrt(1+(slope**2)))
    return x_end, y_end

for filename in os.listdir(png_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(png_folder, filename)
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

        vessel_pixels_x = vessel_pixels[1]
        vessel_pixels_y = vessel_pixels[0]

        # 3차 다항식으로 curve fitting
        params, covariance = curve_fit(third_order_curve, vessel_pixels_x, vessel_pixels_y,
                                p0=(0, 0, 0, 0))  # 초기 추정값은 0으로 설정하거나 필요에 따라 수정
        a_fit, a1_fit, a2_fit, a3_fit = params

        fit_curve_x = np.arange(int(min(vessel_pixels_x)), int(max(vessel_pixels_x)) + 1, 1)  # 정수 간격으로 x값 생성
        fit_curve_y = third_order_curve(fit_curve_x, a_fit, a1_fit, a2_fit, a3_fit)

        # 시각화 및 저장
        plt.figure()
        plt.imshow(segmentation_result, cmap='gray')
        plt.plot(fit_curve_x, fit_curve_y, color=(1, 0, 0, 0.5), linewidth=2)
        plt.axis('off')

        # 미분값 계산
        derivative_values = derivative_third_order_curve(fit_curve_x, a1_fit, a2_fit, a3_fit)

        # 법선 벡터의 기울기 계산
        # 미분값이 0인 경우 예외 처리
        normal_vector_gradients = np.zeros_like(derivative_values)
        non_zero_indices = np.where(derivative_values != 0)
        normal_vector_gradients[non_zero_indices] = -1 / derivative_values[non_zero_indices]


        # .txt 파일로 저장
        output_txt_path = os.path.join(output_folder, filename.replace('.png', '_vector.txt'))
        
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write(f"A0: {a_fit}\n")
            txt_file.write(f"A1: {a1_fit}\n")
            txt_file.write(f"A2: {a2_fit}\n")
            txt_file.write(f"A3: {a3_fit}\n")
            txt_file.write("\n")
            for x, y, gradient in zip(fit_curve_x, fit_curve_y, normal_vector_gradients):
                if gradient == 0:
                    txt_file.write(f"{x} {512-y} \t{np.inf} {np.inf}\t0 (Derivative is 0, Normal Vector Direction is Infinite)\n")
                else:
                    end_x, end_y = get_end(x, y, gradient)
                    txt_file.write(f"{x} {512-y} \t{end_x} {512-end_y}\t{(-1)*gradient}\n")

        seg_image_path = os.path.join(output_folder, filename.replace('.png', '_curve.png'))
        plt.savefig(seg_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print('saved..')
