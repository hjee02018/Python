import os
import glob
import numpy as np
import pydicom
from scipy.ndimage import zoom
import imageio

def add_gaussian_noise(inp, expected_noise_ratio=0.05):
    image = inp.copy()
    if len(image.shape) == 2:
        row, col = image.shape
        ch = 1
    else:
        row, col, ch = image.shape
    mean = 0.
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col)) * expected_noise_ratio
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

def normalize(img):
    arr = img.copy().astype(float)
    M = float(np.max(img))
    if M != 0:
        arr *= 1. / M
    return arr

def preprocess(filename, resize_ratio=0.25):
    ds = pydicom.dcmread(filename)
    img = ds.pixel_array
    img = normalize(zoom(img, resize_ratio))
    img = add_gaussian_noise(img)
    return img

# DICOM 파일이 저장된 디렉토리 경로 설정
dicom_dir = "C:/Users/USER/Desktop/3Dircadb1.1/PATIENT_DICOM"

# DICOM 파일들의 경로 리스트 가져오기
filelist = glob.glob(os.path.join(dicom_dir, "*"))

# DICOM 파일들 전처리하여 PNG 파일로 저장
for dicom_file in filelist:
    print(dicom_file)
    if not dicom_file.endswith(".dcm"):
        new_filename = dicom_file + ".dcm"  # 파일 이름에 .dcm 확장자 추가
        os.rename(dicom_file, new_filename)
        dicom_file = new_filename  # 파일 경로에 .dcm 확장자 추가
        print(dicom_file+" renamed!")
    pp_image = preprocess(dicom_file)
    pp_image = (pp_image * 255).astype(np.uint8)  # convert to uint8
    imageio.imwrite(os.path.splitext(dicom_file)[0] + ".png", pp_image, check_contrast=False)
