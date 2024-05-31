import gradio as gr
import sys
import json
from google.cloud import automl_v1beta1
import numpy as np
import cv2
import imutils
from sklearn.cluster import KMeans
import os
from scipy.spatial import distance as dist
import time
from PIL import Image, ImageDraw



def image_classifier(img):
        import sys
        import json
        from google.cloud import automl_v1beta1
        import numpy as np
        import cv2
        import imutils
        from sklearn.cluster import KMeans
        import os
        from scipy.spatial import distance as dist
        import time
        from PIL import Image, ImageDraw

        debug = False
        debug_display = False
        write_images = False
        def display(name,img,convert=False):
            img_cp = img.copy()
            if convert:
                img_cp = img_cp * 255
                img_cp = img_cp.astype(np.uint8)
            cv2.imshow(name,img_cp)
            cv2.waitKey(0)

        def write_exit(name,img,convert=False):
            if convert:
                img_cp = img.copy()
                img_cp = img_cp*255
                img_cp = img_cp.astype(np.uint8)
                cv2.imwrite('pictures/' + name + '.jpg', img_cp)
            else: cv2.imwrite(name+'.jpg',img)
            exit()

        def write(name,img,convert=False):
            if convert:
                img_cp = img.copy()
                img_cp = img_cp*255
                img_cp = img_cp.astype(np.uint8)
                cv2.imwrite('pictures/yellow_pill/' + name + '.jpg', img_cp)
            else: cv2.imwrite('pictures/yellow_pill/' + name+'.jpg',img)

        def mse(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])
            return err
        
        def detect_color(image):
            #image = cv2.imread(image)
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

        dominant_color = detect_color(img)

        rgb_color = dominant_color
        print(rgb_to_color(rgb_color))

        
        import cv2
        import easyocr
        import pandas as pd
        #img = cv2.imread('pill_image.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reader = easyocr.Reader(["en"], verbose=False)
        result = reader.readtext(gray, paragraph=False)
        imprints = ['-'.join(text.split()) for _, text, _ in result if text.strip()]
        imprints_joined = ' '.join(imprints)
        print(imprints_joined)

        from bs4 import BeautifulSoup
        import sys
        import requests
        import json
        import urllib.parse

        # Replace with the actual URL of the drugs.com search page
        root = "https://www.drugs.com/imprints.php?"  # Example URL
        imprint = imprints_joined
        color_input = rgb_color



        # Construct URL with proper encoding
        url = f"{root}imprint={imprint}"
        page = requests.get(url)
        #soup =  BeautifulSoup(page.content, 'html.parser')
        print(url)




        if page.status_code == 200:  # Check for successful response
            soup = BeautifulSoup(page.content, 'html.parser')

        # Find relevant elements based on drugs.com's structure (replace with actual selectors)
        drug_results = soup.find('a', class_='ddc-btn ddc-btn-small', href=True)  # Get href if found
        if drug_results:
            link_url = drug_results['href']
            print(f"Link URL: {drug_results['href']}")
        else:
            print("No matching anchor tag found.")


        from bs4 import BeautifulSoup
        import requests


        # Construct the full URL (assuming a base URL)
        base_url = "https://www.drugs.com"  # Replace with the actual base URL if needed
        full_url = base_url + link_url
        print(full_url)

        # Open the linked webpage using requests
        response = requests.get(full_url)

        # Check for successful response (status code 200)
        if response.status_code == 200:
            # Parse the content of the linked webpage
            soup = BeautifulSoup(response.content, 'html.parser')



        # Example: Extract text from all paragraphs
            #drug_sublink = soup.find('div', class_='ddc-form-actions ddc-form-actions-stacked', href=True)  # Get href if found
            parent_element = soup.find('div', class_='ddc-form-actions ddc-form-actions-stacked')
            drug_sublink = parent_element.find('a', class_='ddc-btn', href=True)
            print(drug_sublink)


            

            if drug_sublink:
                #l_url = drug_sublink['href']
                l_url = drug_sublink['href']
                print(f"sub Link URL: {drug_sublink['href']}")
            else:
                print("No matching anchor tag found.")


            b_url = "https://www.drugs.com"  # Replace with the actual base URL if needed
            f_url = b_url + l_url
            print(f_url)

                # Open the linked webpage using requests
            response = requests.get(f_url)

            # Check for successful response (status code 200)
            if response.status_code == 200:
            # Parse the content of the linked webpage
                soup = BeautifulSoup(response.content, 'html.parser')    
            
                
                text_paragraphs = soup.find_all('p', class_= "drug-subtitle")
                for paragraph in text_paragraphs:
                    
                    print(paragraph.text.strip())
                    new = paragraph.text.strip()
            else:
                print(f"Error: Failed to retrieve linked page. Status code: {response.status_code}")
            

                
        else:
            print(f"Error: Failed to retrieve linked page. Status code: {response.status_code}")



        return new


#demo = gr.Interface(fn=image_classifier,inputs="image",outputs="textbox",title="Drug Pill Identifier")
#demo.launch(share=True)
