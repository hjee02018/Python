import cv2
import numpy as np

# 이미지를 불러오고 그레이스케일로 변환합니다.
image = cv2.imread('input.PNG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT 생성
sift = cv2.SIFT_create()

# SIFT 특징점 검출
keypoints = sift.detect(image, None)


# 특징점 간의 거리를 계산하고, 가장 거리가 먼 두 개의 특징점 선택
max_distance = 0
selected_keypoints = None

for i, keypoint1 in enumerate(keypoints):
    for j, keypoint2 in enumerate(keypoints):
        if i != j:
            distance = np.sqrt((keypoint1.pt[0] - keypoint2.pt[0])**2 + (keypoint1.pt[1] - keypoint2.pt[1])**2)
            if distance > max_distance:
                max_distance = distance
                selected_keypoints = (keypoint1, keypoint2)

# 선택된 두 개의 특징점 좌표를 출력
x1, y1 = map(int, selected_keypoints[0].pt)
x2, y2 = map(int, selected_keypoints[1].pt)
print(f"첫 번째 특징점 좌표: ({x1}, {y1})")
print(f"두 번째 특징점 좌표: ({x2}, {y2})")

# 결과 이미지를 표시하고 선택된 두 개의 특징점을 표시
result_image = cv2.drawKeypoints(image, selected_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT Keypoints', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


rect_x1 = min(x1, x2)
rect_y1 = min(y1, y2)
rect_x2 = max(x1, x2)
rect_y2 = max(y1, y2)

cropped_image = image[rect_y1:rect_y2, rect_x1:rect_x2]
cv2.imwrite('cropped_result.PNG', cropped_image)
