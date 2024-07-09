import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(image, 5)
    # Define a lower and upper color range for the lava (you can adjust these values)
    lower_lava = np.array([0, 100, 122], dtype="uint8")
    upper_lava = np.array([20, 255, 255], dtype="uint8")

    # Create a mask for the lava region
    mask = cv2.inRange(blur, lower_lava, upper_lava)
    # Apply Otsu's thresholding to the mask
    _, thresholded = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find contours in the thresholded mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    max_contour = max(contours, key=cv2.contourArea)

    # Create a mask to fill the largest contour with white
    mask = np.zeros_like(thresholded)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    # Set the background as black
    result = np.zeros_like(image)

    # Color the largest contour white
    result[mask == 255] = [255, 255, 255]

    return result