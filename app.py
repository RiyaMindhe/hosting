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
        
        #def custom_function(image):
          #return image
          
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

        def clustering(pixels,h,w):
            c = 3
            cluster = KMeans(n_clusters=c).fit(pixels)
            colors = cluster.cluster_centers_  # the cluster centers are the dominant colors
            predictions = cluster.predict(pixels)

            new_img = np.zeros((h, w, 3))
            counter = 0
            for i in range(h):
                for j in range(w):
                    new_img[i][j] = colors[predictions[counter]][::-1]
                    counter += 1
            if debug_display: display('K-Means-Color', new_img, convert=True)
            if write_images: write('K-Means-Color',new_img,convert=True)

            freq = {}
            for l in predictions:
                ll = tuple(colors[l])
                if ll in freq: freq[ll] += 1
                else: freq[ll] = 1
            color_rgb = [(x * 255, y * 255, z * 255) for [x, y, z] in colors]

            f = open("helper_files/color.txt", "r")
            color_values = [tuple((i.split(":")[0], eval(i.split(" ")[1]))) for i in (f.read()).split("\n")]
            f.close()

            for j in color_rgb:
                minDist = (np.inf, None)
                for (i, row) in enumerate(color_values):
                    d = dist.euclidean(row[1], j)
                    if d < minDist[0]: minDist = (d, row[0])

                if minDist[1] == 'Black':
                    check = (j[0] / 255, j[1] / 255, j[2] / 255)
                    if check in freq: del freq[check]
                    color_rgb.remove(j)
                    break

            colors = ["blue", "brown", "gray", "green", "orange", "purple", "pink", "red", "turquoise", "white", "yellow"]
            color_values = {"blue": [], "brown": [], "gray": [], "green": [], "orange": [], "purple": [], "pink": [], "red": [],
                            "turquoise": [], "white": [], "yellow": []}
            for color in colors:
                f2 = open('helper_files/' + color + ".txt", "r")
                for i in f2: color_values[color].append(eval(i))
                f2.close()

            classified = []
            for c in color_rgb:
                check = (c[0] / 255, c[1] / 255, c[2] / 255)
                all = []
                for color in colors:
                    for i in color_values[color]:
                        if i:
                            d = dist.euclidean(c, i)
                            all.append((d, color))
                all.sort(key=lambda x: x[0])
                all = all[:5]
                final = {}
                for i in all:
                    if i[1] in final:
                        final[i[1]][0] += 1
                        final[i[1]][1] += -i[0]
                    else: final[i[1]] = [1, -i[0], i[1]]

                final = list(final.values())
                final.sort(key=lambda x: (x[0], x[1]), reverse=True)
                if final[0][2] not in classified: classified.append((final[0][2],c))
                if check in freq:
                    freq[final[0][2]] = freq[check]
                    del freq[check]

            if len(classified) > 1:
                if freq[classified[0][0]] <= 0.75 * freq[classified[1][0]]: del classified[0]
                elif freq[classified[1][0]] <= 0.75 * freq[classified[0][0]]: del classified[1]
            return classified



        img2 = img
        #img2 = cv2.imread(sys.argv[1])
        if debug_display: display('Original',img2)
        img = img2.copy()
        img3 = img2.copy()
        h,w = img.shape[:2]

        # Grabcut
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (5,5,img.shape[1]-5,img.shape[0]-5)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        if debug_display: display('first_grabcut',img)
        if write_images: write('first_grabcut',img)

        # Output Color
        pixels = []
        for i in range(h):
            for j in range(w): pixels.append(img[i][j][::-1]/255)
        color_output = list(set(col[0] for col in clustering(pixels,h,w)))
        print('Color: ', color_output)

        
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
        color_input = color_output

        color_to_number = {
        
        "beige": 14,
        "black": 73,
        "blue": 1,
        "brown": 2,
        "clear": 3,
        "gold": 4,
        "gray": 5,
        "green": 6,
        "maroon": 44,
        "orange": 7,
        "peach": 74,
        "pink": 8,
        "purple": 9,
        "red": 10,
        "tan": 11,
        "white": 12,
        "yellow": 13,			
        "beige&red": 69,
        "black&green": 55,
        "black&teal": 70,
        "black&yellow": 48,
        "blue&brown": 52,
        "blue&gray": 45,
        "blue&green": 75,
        "blue&orange": 71,
        "blue&peach": 53,
        "blue&pink": 34,
        "blue&white": 19,
        "blue&white specks": 26,
        "blue&yellow": 21,
        "brown&clear": 47,
        "brown&orange": 54,
        "brown&peach": 28,
        "brown&red": 16,
        "brown&white": 57,
        "brown&yellow": 27,
        "clear&green": 49,
        "dark&light green": 46,
        "gold&white": 51,
        "gray&peach": 61,
        "gray&pink": 39,
        "gray&red": 58,
        "gray&white": 67,
        "gray&yellow": 68,
        "green&orange": 65,
        "green&peach": 63,
        "green&pink": 56,
        "green&purple": 43,
        "green&turquoise": 62,
        "green&white": 30,
        "green&yellow": 22,
        "lavender&white": 42,
        "maroon&pink": 40,
        "orange&turquoise": 50,
        "orange&white": 64,
        "orange&yellow": 23,
        "peach&purple": 60,
        "peach&red": 66,
        "peach&white": 18,
        "pink&purple": 15,
        "pink&red specks": 37,
        "pink&turquoise": 29,
        "pink&white": 25,
        "pink&yellow": 72,
        "red&turquoise": 17,
        "red&white": 35,
        "red&yellow": 20,
        "tan&white": 33,
        "turquoise&white": 59,
        "turquoise&yellow": 24,
        "white&blue specks": 32,
        "white&red specks": 41,
        "white&yellow": 38,
        "yellow&gray": 31,
        "yellow&white": 36
            
            
        }


        def get_color_number(color):

        # Convert input color to lowercase for case-insensitive matching
                for color in color_input:
                    color = color.lower()  # Convert to lowercase
                    #print(color_lowercase)

                if color in color_to_number:
                    col_to_num = color_to_number[color]
                    return col_to_num
                else:
                    return None



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


demo = gr.Interface(fn=image_classifier,inputs="image",outputs="textbox",title="Drug Pill Identifier")
demo.launch(share=True)



