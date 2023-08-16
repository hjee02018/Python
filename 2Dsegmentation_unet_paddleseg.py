import numpy as np
import paddle
from paddleseg.models import UNet
import matplotlib.pyplot as plt
from PIL import Image

# Pre-trained 모델 불러오기 및 설정
model = UNet(num_classes=2)  # 클래스 수에 맞게 설정
model_path = 'model.pdparams'  # 다운로드한 Pre-trained 모델의 경로
model.set_dict(paddle.load(model_path))
model.eval()

# 입력 데이터 전처리
image_path = 'test.jpg'  # 입력 이미지 파일 경로
image = Image.open(image_path).convert('RGB')  # 컬러 이미지로 변환
image = np.array(image)
image = image.astype(np.float32) / 255.0  # 이미지를 [0, 1] 범위로 정규화
image = np.transpose(image, (2, 0, 1))  # 채널 순서 변경 (H, W, C) -> (C, H, W)
image = np.expand_dims(image, axis=0)  # 배치 차원 추가

input_data = paddle.to_tensor(image)

# 추론 수행
with paddle.no_grad():
    output = model(input_data)

# 모델 출력을 확장하고 텐서 형태로 변환
output = [paddle.to_tensor(output_item) for output_item in output]

# 세그멘테이션 결과에서 클래스 예측값 추출
argmax_output = paddle.argmax(output[0], axis=1)
segmentation_result = argmax_output.numpy()[0]  # 배치 차원 제거

# 이제 segmentation_result를 시각화하거나 저장할 수 있습니다.

# 클래스에 따라 색상을 지정합니다. (0: 배경, 1: 혈관)
colors = [
    [0, 0, 0],   # 배경: 검은색
    [255, 0, 0]  # 혈관: 빨간색
]

# 클래스에 따라 색상을 적용한 이미지를 생성합니다.
colored_segmentation = np.zeros((segmentation_result.shape[0], segmentation_result.shape[1], 3), dtype=np.uint8)
for class_idx, color in enumerate(colors):
    mask = segmentation_result == class_idx
    for c in range(3):  # 각 채널에 대해 색상을 적용
        colored_segmentation[mask, c] = color[c]

# 시각화
plt.imshow(colored_segmentation)
plt.axis('off')
plt.show()
