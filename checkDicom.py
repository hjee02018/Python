import pydicom

# DICOM 파일 경로
dcm_path = 'sample.dcm'

# DICOM 파일 읽기
dcm = pydicom.dcmread(dcm_path)

# 슬라이스 개수 확인
num_slices = dcm.NumberOfFrames if hasattr(dcm, 'NumberOfFrames') else 1

# 출력
if num_slices == 1:
    print("이 DICOM 파일은 2D 이미지입니다.")
else:
    print(f"이 DICOM 파일은 {num_slices}개의 슬라이스를 포함하는 3D 데이터입니다.")
