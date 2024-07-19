import cv2
import numpy as np
from skimage import transform, img_as_ubyte
def rotateImage(Image, angle: float):
    newImage = Image.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle+.05, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    newImage = cv2.warpAffine(newImage, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def solution(image_path):
    img = cv2.imread(image_path)
    inv_gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(inv_gray, 150, 255, cv2.THRESH_BINARY)

    optimal_angle = 0
    max_sum = 0
    
    for angle in range(180):
        rotated_img = img_as_ubyte(transform.rotate(binary_img, angle, resize=True))
        row_sums = np.sum(rotated_img, axis=1)
        
        if max_sum <= np.max(row_sums):
            max_sum = np.max(row_sums)
            optimal_angle = angle

    rotated_img = img_as_ubyte(transform.rotate(binary_img, optimal_angle, resize=True))
    row_sums = np.sum(rotated_img, axis=1)
    max_row_idx = np.argmax(row_sums)
    
    flip_check = np.sum(row_sums[max_row_idx-4:max_row_idx-1]) > np.sum(row_sums[max_row_idx+1:max_row_idx+4])
    if flip_check:
        optimal_angle -= 180
    final_img=rotateImage(img, optimal_angle)
    
    return final_img
