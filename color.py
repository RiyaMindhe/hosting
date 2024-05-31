import cv2
import numpy as np

def detect_color(image_path):
    image = cv2.imread(image_path)
    print("Image shape:", image.shape)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = np.uint8(centers[0])   
    return dominant_color

def rgb_to_color(rgb):
    red, green, blue = rgb
    red_threshold = 127
    green_threshold = 127
    blue_threshold = 127
    if red > red_threshold and green < green_threshold and blue < blue_threshold:
        return 'Red'
    elif red < red_threshold and green > green_threshold and blue < blue_threshold:
        return 'Green'
    elif red < red_threshold and green < green_threshold and blue > blue_threshold:
        return 'Blue'
    elif red > red_threshold and green > green_threshold and blue < blue_threshold:
        return 'Yellow'
    elif red > red_threshold and green < green_threshold and blue > blue_threshold:
        return 'Magenta'
    elif red < red_threshold and green > green_threshold and blue > blue_threshold:
        return 'Cyan'
    elif red > red_threshold and green > green_threshold and blue > blue_threshold:
        return 'White'
    else:
        return 'Black'


image_path = r"D:\Riya\justdata\Google Chrome ( Final Year )\final dataset\val\1\24 HR dexmethylphenidate hydrochloride 15 MG Extended Release Oral Capsule.jpg"
dominant_color = detect_color(image_path)

rgb_color = dominant_color
print(rgb_to_color(rgb_color))  

