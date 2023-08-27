import os
import numpy as np
import paddle
from paddleseg.models import UNet
from PIL import Image
import sys
import pydicom

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
    shape = ds.pixel_array.shape
    image_2d = ds.pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)
    if len(image_2d_scaled.shape) > 2:
        image_2d_scaled = image_2d_scaled[:, :, 0]  # Take the first channel (grayscale)
    img = Image.fromarray(image_2d_scaled, mode='L')  # 'L' mode indicates grayscale
    output_png_path = os.path.join(png_folder, os.path.splitext(dicom_file)[0] + '.png')
    img.save(output_png_path)

print("Conversion complete.")

# Segmentation

# Pre-trained 모델 불러오기 및 설정
model = UNet(num_classes=2)  # 클래스 수에 맞게 설정
model_path = 'model.pdparams'  # 다운로드한 Pre-trained 모델(unet)의 경로
model.set_dict(paddle.load(model_path))
model.eval()

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

        output_path = os.path.join(output_folder, filename)
        result_image = Image.fromarray(colored_segmentation)
        result_image.save(output_path)
        print(f"Segmentation result saved at '{output_path}'")
