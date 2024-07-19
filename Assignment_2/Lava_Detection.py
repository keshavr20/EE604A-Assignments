import cv2
import numpy as np

def solution(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred_img = cv2.medianBlur(img_rgb, 5)
    lava_lower_bound = np.array([150, 0, 0], dtype="uint8")
    lava_upper_bound = np.array([255, 255, 110], dtype="uint8")
    lava_mask = cv2.inRange(blurred_img, lava_lower_bound, lava_upper_bound)
    _, binary_img = cv2.threshold(lava_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morphed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)))
    contours, _ = cv2.findContours(morphed_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    approx_contour = cv2.approxPolyDP(largest_contour, 0.0008 * cv2.arcLength(largest_contour, True), True)
    final_mask = np.zeros_like(binary_img)
    cv2.drawContours(final_mask, [approx_contour], -1, 255, thickness=cv2.FILLED)
    final_result = np.zeros_like(img_rgb)
    final_result[final_mask == 255] = [255, 255, 255]
    return final_result
