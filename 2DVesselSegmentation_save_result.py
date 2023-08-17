import os
import numpy as np
import paddle
from paddleseg.models import UNet
from PIL import Image

# Pre-trained 모델 불러오기 및 설정
model = UNet(num_classes=2)  # 클래스 수에 맞게 설정
model_path = 'model.pdparams'  # 다운로드한 Pre-trained 모델(unet)의 경로
model.set_dict(paddle.load(model_path))
model.eval()

input_dir = r'data/png/F208-LAD(3)'
output_dir = 'data/output/F208-LAD(3)'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 세그멘테이션 수행 및 결과 저장
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
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

        output_path = os.path.join(output_dir, filename)
        result_image = Image.fromarray(colored_segmentation)
        result_image.save(output_path)
        print(f"Segmentation result saved at '{output_path}'")
