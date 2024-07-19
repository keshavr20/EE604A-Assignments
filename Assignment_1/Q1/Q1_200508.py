import cv2
import numpy as np

def is_color_in_range(color, target_color, threshold=5):
    return all(abs(color[i] - target_color[i]) <= threshold for i in range(3))

# Usage
def solution(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bottom_left = None
    bottom_right = None
    top_right = None
    top_left = None

    half_width = width // 2
    half_height = height // 2

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            for point in approx:
                x, y = point.ravel()
                if x < half_width and y < half_height:
                    bottom_left = (x, y)
                elif x > half_width and y < half_height:
                    bottom_right = (x, y)
                elif x > half_width and y > half_height:
                    top_right = (x, y)
                elif x < half_width and y > half_height:
                    top_left = (x, y)
            break

    corner_colors = [image[y, x] for x, y in [bottom_left, bottom_right, top_right, top_left]]
    reference_image = np.ones((600, 600, 3), dtype=np.uint8) * 255
    saffron = (51, 153, 255)
    green = (0, 128, 0)
    navy_blue = (255, 0, 0)
    cv2.rectangle(reference_image, (0, 0), (600, 200), green, -1)
    cv2.rectangle(reference_image, (0, 400), (600, 600), saffron, -1)
    center = (300, 300)
    radius = 100
    cv2.circle(reference_image, center, radius, navy_blue, 2)
    for angle in range(0, 360, 15):
        theta = np.radians(angle)
        x1 = int(center[0] + radius * np.cos(theta))
        y1 = int(center[1] + radius * np.sin(theta))
        x2 = center[0]
        y2 = center[1]
        cv2.line(reference_image, (x1, y1), (x2, y2), navy_blue, 1)

    reference_image_2 = cv2.rotate(reference_image, cv2.ROTATE_90_CLOCKWISE)
    reference_image_3 = cv2.rotate(reference_image_2, cv2.ROTATE_90_CLOCKWISE)
    reference_image_4 = cv2.rotate(reference_image_3, cv2.ROTATE_90_CLOCKWISE)

    if is_color_in_range(corner_colors[0], green) and is_color_in_range(corner_colors[1], green):
        image = reference_image
    elif is_color_in_range(corner_colors[0], saffron) and is_color_in_range(corner_colors[1], green):
        image = reference_image_2
    elif is_color_in_range(corner_colors[0], green) and is_color_in_range(corner_colors[1], saffron):
        image = reference_image_4
    elif is_color_in_range(corner_colors[0], saffron) and is_color_in_range(corner_colors[1], saffron):
        image = reference_image_3
    
    return image
