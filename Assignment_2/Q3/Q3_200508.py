import cv2
import numpy as np

class ImageCropper:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def convert_to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def detect_and_compute(self, img):
        feature_detector = cv2.ORB_create()
        return feature_detector.detectAndCompute(img, None)

    def match_points(self, desc_img1, desc_img2):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return matcher.match(desc_img1, desc_img2)

    def filter_matches(self, matched_pts, threshold_value=1000):
        return [point for point in matched_pts if point.distance < threshold_value]

    def find_homography(self, img1_pts, img2_pts):
        return cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC)

    def warp_perspective(self, img, homography):
        img_height, img_width = np.shape(self.img1)[:2]
        return cv2.warpPerspective(img, homography, (img_width, img_height))

    def chk_ref_img(self):
        try: 
            gray_img1 = self.convert_to_grayscale(self.img1)
            gray_img2 = self.convert_to_grayscale(self.img2)

            kps_img1, desc_img1 = self.detect_and_compute(gray_img1)
            kps_img2, desc_img2 = self.detect_and_compute(gray_img2)

            matched_pts = self.match_points(desc_img1, desc_img2)
            matched_pts = sorted(matched_pts, key=lambda x: x.distance)

            valid_matches = self.filter_matches(matched_pts)

            img1_pts = np.float32([kps_img1[match.queryIdx].pt for match in valid_matches]).reshape(-1, 1, 2)
            img2_pts = np.float32([kps_img2[match.trainIdx].pt for match in valid_matches]).reshape(-1, 1, 2)

            homography, _ = self.find_homography(img1_pts, img2_pts)

            processed_image = self.warp_perspective(self.img2, homography)

            return processed_image
        
        except Exception as error:
            return self.img2

class ImageMatcher:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_kp_desc(self, gray_img):
        orb = cv2.ORB_create()
        return orb.detectAndCompute(gray_img, None)

    def get_matches(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return bf.match(desc1, desc2)

    def chk_img_match(self):
        try: 
            gray1 = self.to_gray(self.img1)
            gray2 = self.to_gray(self.img2)

            kp1, desc1 = self.get_kp_desc(gray1)
            kp2, desc2 = self.get_kp_desc(gray2)

            matches = self.get_matches(desc1, desc2)
            sorted_matches = sorted(matches, key=lambda x: x.distance)
        except Exception as e:
            return 0
    
        return len(sorted_matches)

def process_image(img1, img2):
    img2_rsd = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img2_crpd = ImageCropper(img1, img2).chk_ref_img()
    img2_crpd_rsd = ImageCropper(img1, img2_crpd).chk_ref_img()
    img2_rsd_crpd = ImageCropper(img1, img2_rsd).chk_ref_img()

    match_scores = [
        ImageMatcher(img1, img2).chk_img_match(),
        ImageMatcher(img1, img2_rsd).chk_img_match(),
        ImageMatcher(img1, img2_crpd).chk_img_match(),
        ImageMatcher(img1, img2_crpd_rsd).chk_img_match(),
        ImageMatcher(img1, img2_rsd_crpd).chk_img_match()
    ]
    match_scores.sort(reverse=True)

    return match_scores[0]

def solution(audio_path):
    image1 = np.array(real_image1, dtype=np.uint8)
    image2 = cv2.imread(audio_path)
    match_score = process_image(image1, image2)
    class_name = "real" if match_score >= 425 else "fake"
    if class_name == "fake":
        image1 = np.array(real_image2, dtype=np.uint8)
        match_score = process_image(image1, image2)
        class_name = "real" if match_score >= 350 else "fake"

    return class_name


