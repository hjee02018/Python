import os
import numpy as np
import paddle
from paddleseg.models import UNet
from PIL import Image
import sys
import pydicom
import cv2

import glob
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# get argument from user
arg = sys.argv[1]
dicom_folder = f'data/dicom/{arg}'
png_folder = f'data/png/{arg}'
output_folder = f'data/output/{arg}'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)
# Ensure output directory exists
os.makedirs(png_folder, exist_ok=True)

# ADD : DCM to PNG conversion
dicom_files = [filename for filename in os.listdir(dicom_folder) if filename.endswith('.dcm')]
for dicom_file in dicom_files:
    ds = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))
    
    # Check if the DICOM is a multiframe image
    num_slices = ds.NumberOfFrames if hasattr(ds, 'NumberOfFrames') else 1
    if num_slices > 1:
        # Multiframe case: Process each frame and save as PNG
        for idx in range(num_slices):
            print(num_slices)
            frame_data = ds[idx]
            image_2d = frame_data.pixel_array.astype(float)
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            image_2d_scaled = np.uint8(image_2d_scaled)
            if len(image_2d_scaled.shape) > 2:
                image_2d_scaled = image_2d_scaled[:, :, 0]  # Take the first channel (grayscale)
            img = Image.fromarray(image_2d_scaled, mode='L')  # 'L' mode indicates grayscale
            
            # Append slice index to the output PNG file name
            output_png_path = os.path.join(png_folder, f"{os.path.splitext(dicom_file)[0]}_{idx}.png")
            img.save(output_png_path)
    else:
        # Single image case: Process the single image and save as PNG
        image_2d = ds.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
        if len(image_2d_scaled.shape) > 2:
            image_2d_scaled = image_2d_scaled[:, :, 0]  # Take the first channel (grayscale)
        img = Image.fromarray(image_2d_scaled, mode='L')  # 'L' mode indicates grayscale
        
        # Save the single image as PNG
        output_png_path = os.path.join(png_folder, os.path.splitext(dicom_file)[0] + '.png')
        img.save(output_png_path)


# Segmentation

# Pre-trained 모델 불러오기 및 설정
model = UNet(num_classes=2)  # 클래스 수에 맞게 설정
model_path = 'model.pdparams'  # 다운로드한 Pre-trained 모델(unet)의 경로
model.set_dict(paddle.load(model_path))
model.eval()

# 8차 다항식 함수 정의
def eighth_order_curve(x, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5 + a6 * x**6 + a7 * x**7 + a8 * x**8


# 세그멘테이션 수행 및 결과 저장
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
        
        # 혈관 픽셀 좌표 추출
        vessel_pixels_x = vessel_pixels[1]
        vessel_pixels_y = vessel_pixels[0]

        # 8차 다항식으로 curve fitting
        params, covariance = curve_fit(eighth_order_curve, vessel_pixels_x, vessel_pixels_y,
                                p0=(0, 0, 0, 0, 0, 0, 0, 0, 0))  # 초기 추정값은 0으로 설정하거나 필요에 따라 수정
        a_fit, a1_fit, a2_fit, a3_fit, a4_fit, a5_fit, a6_fit, a7_fit, a8_fit = params

        # Fitting된 곡선 좌표 계산
        fit_curve_x = np.linspace(min(vessel_pixels_x), max(vessel_pixels_x), 100)
        fit_curve_y = eighth_order_curve(fit_curve_x, a_fit, a1_fit, a2_fit, a3_fit, a4_fit, a5_fit, a6_fit, a7_fit, a8_fit)

        
        # 시각화 및 저장
        plt.figure()
        plt.imshow(segmentation_result, cmap='gray')
        plt.plot(fit_curve_x, fit_curve_y, color='red', linewidth=2)
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, filename.replace('_6.png', '_6_curve.png')), bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Segmentation result with curve saved")
