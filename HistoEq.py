import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 히스토그램 계산
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 누적 분포 함수 생성
    cdf = hist.cumsum()

    # cdf의 최소값이 0이 되지 않도록 보정
    cdf_min = np.ma.masked_equal(cdf, 0)
    cdf_min = (cdf_min - cdf_min.min()) * 255 / (cdf_min.max() - cdf_min.min())
    cdf = np.ma.filled(cdf_min, 0).astype('uint8')

    # 이미지에 적용된 누적 분포 함수를 이용하여 강도 변환
    equalized_image = cdf[image]
    # 결과 이미지 시각화    
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])

    plt.show()

# 이미지 경로 설정
image_path = 'smp.png'

# 히스토그램 평활화 함수 호출
histogram_equalization(image_path)
