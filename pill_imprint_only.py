"""
Before start:
export GOOGLE_APPLICATION_CREDENTIALS=helper_files/research-pill-google_service_account_key.json
export PROJECT_ID=research-pill
"""
#input at line 132

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

        #show temp image
        #import matplotlib.pyplot as plt
        #image_after_grabcut = plt.imread(img)
        # Display the image
        #plt.imshow(img)
        #plt.show()

        ##imprint code

        '''import pandas as pd
        import cv2
        import easyocr
        #img = cv2.imread('temp.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise=cv2.medianBlur(gray,3)
        thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        reader = easyocr.Reader(["en"], verbose=False)
        result = reader.readtext(img,paragraph="False")
        df=pd.DataFrame(result)
        df[1] = df[1].str.strip()
        print(df[1])'''

        import cv2
        import easyocr
        import pandas as pd
        #img = cv2.imread('pill_image.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reader = easyocr.Reader(["en"], verbose=False)
        result = reader.readtext(gray, paragraph=False)
        imprints = [text.strip() for _, text, _ in result if text.strip()]
        imprints_without_spaces = ''.join(imprints)
        print(imprints_without_spaces)
        

        



        '''# K-Means
        pixels = []
        for i in range(h):
            for j in range(w): pixels.append(img[i][j]/255)
        c = 4
        cluster = KMeans(n_clusters=c).fit(pixels)
        labels = cluster.predict(pixels)
        colors = cluster.cluster_centers_
        new_img = np.zeros((h, w, 3))
        counter = 0
        for i in range(h):
            for j in range(w):
                new_img[i][j] = colors[labels[counter]]
                counter += 1
        if debug_display: display('K-Means-pre',new_img)
        if write_images: write('K-Means-pre',new_img, convert=True)

        # K-Means Post Processing
        edge_pixels = {}
        w_in = [x for x in range(5,w-5)]
        h_in = [x for x in range(5,h-5)]
        for i in range(h):
            for j in range(w):
                if i not in h_in:
                    if tuple(new_img[i][j]) not in edge_pixels: edge_pixels[tuple(new_img[i][j])] = 0
                    edge_pixels[tuple(new_img[i][j])] += 1
                if j not in w_in:
                    if tuple(new_img[i][j]) not in edge_pixels: edge_pixels[tuple(new_img[i][j])] = 0
                    edge_pixels[tuple(new_img[i][j])] += 1

        to_delete = [e for e in edge_pixels if edge_pixels[e]<500]
        for t in to_delete: del edge_pixels[t]
        edge_pixels = set(edge_pixels.keys())
        for i in range(h):
            for j in range(w):
                if tuple(new_img[i][j]) in edge_pixels:
                    new_img[i][j] = [0,0,0]
        if debug_display: display('K-Means-post',new_img)
        if write_images: write('K-Means-post',new_img, convert=True)

        img = new_img * 255
        img = img.astype(np.uint8)

        # Contour Extraction
        img[np.where((img!=[0,0,0]).all(axis=2))] = [255,255,255]
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        final = []
        for c in cnts:
            if len(final)==0:
                final = c
            elif len(c) > len(final):
                final = c

        x, y, w, h = cv2.boundingRect(final)
        white = np.zeros((h+5,w+5),np.uint8)
        white[white == 0] = 255
        x_n = [c[0][0]-x+3 for c in final]
        y_n = [c[0][1]-y+3 for c in final]
        arr = [[x_n[i], y_n[i]] for i in range(len(x_n))]
        ctr = np.array(arr).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(white, [ctr], -1, (0, 0, 0), 8)
        if debug_display: display('Outline',white)
        if write_images: write('Outline',white)

        # Storing Image temporarily to make it an actual (3-D) image
        cv2.imwrite('temp.jpg',white)
        white = cv2.imread('temp.jpg')



        # Comparing against templates
        shapes = ['bullet', 'capsule', 'diamond', 'double-circle', 'freeform', 'hexagon', 'octagon', 'oval', 'pentagon',
                'rectangle', 'round', 'semi-circle', 'square', 'tear', 'trapezoid', 'triangle']
        groups = [['capsule', 'diamond', 'double-circle', 'oval', 'tear', 'trapezoid'], ['hexagon', 'pentagon', 'round', 'semi-circle', 'square', 'trapezoid', 'triangle'], ['octagon', 'rectangle', 'semi-circle'], ['freeform'], ['bullet']]
        probability = {'round': 2032, 'oval': 1476, 'capsule': 738, 'triangle': 30, 'rectangle': 27, 'diamond': 16, 'pentagon': 15, 'tear': 13, 'hexagon': 12, 'square': 12, 'bullet': 6, 'semi-circle': 5, 'trapezoid': 3, 'freeform': 2, 'octagon': 2, 'double-circle': 3}
        template_folder = './templates/'
        img = imutils.resize(white, width=300)
        h,w = img.shape[:2]
        compare,compare_dict,compare2 = [], {}, []
        for shape in shapes:
            temp = cv2.imread(template_folder+shape+'.png')
            template = temp.copy()

            template = imutils.resize(template, width=300)
            h_t, w_t = template.shape[:2]
            if h >= h_t*2 or h_t >= h*2:
                compare.append((99999999,shape))
            else:
                w_i, h_i = min(w, w_t), min(h, h_t)
                result = mse(img[:h_i,:w_i],template[:h_i,:w_i])
                compare.append((result, shape))
                compare_dict[shape] = result
        compare = [x for x in compare if x[0]!=99999999]
        compare.sort(key=lambda x:x[0])
        if debug: print('Shape Recog: Compare all 1', compare)
        ans = compare[0][1]
        for g in groups:
            if ans in g:
                for gg in g:
                    if gg in compare_dict:
                        result = compare_dict[gg] * (1-probability[gg]/4400)
                        compare2.append((result,gg))
        compare2.sort(key=lambda x:x[0])
        compare = compare2
        if debug: print('Shape Recog: Compare all 2', compare)
        os.remove('temp.jpg')

        imprint_json, data_json, color_json, shape_json = open('helper_files/imprints.json'), open('helper_files/data.json'), open('helper_files/colors.json'), open('helper_files/shapes.json')
        imprints, colors, shapes = json.load(imprint_json), json.load(color_json), json.load(shape_json)
        data = json.load(data_json)
        links = open('links.txt','w')

        # Output Shape
        shapes_output = [compare[0][1]]
        if 'oval' in shapes_output and 'capsule' not in shapes_output: shapes_output.append('capsule')
        if 'capsule' in shapes_output and 'oval' not in shapes_output: shapes_output.append('oval')
        print('Shape:', shapes_output)
        pills_shapes = set()
        for cc in shapes_output:
            if not pills_shapes: pills_shapes = set(shapes[cc])
            else: pills_shapes = pills_shapes.union(set(shapes[cc]))
        if debug: print('Pills after shape: ', pills_shapes)

        pills_color = set()
        for cc in color_output:
            if not pills_color: pills_color = set(colors[cc])
            else: pills_color = pills_color.intersection(set(colors[cc]))
        if debug: print('Pills after Color: ', pills_color)

        # Shape and Color Intersection
        pills = pills_shapes.intersection(pills_color)
        if debug: print('Pills after shape and color: ', pills)'''






        from bs4 import BeautifulSoup
        import sys
        import requests
        import json
        import urllib.parse

        # Replace with the actual URL of the drugs.com search page
        root = "https://www.drugs.com/imprints.php?"  # Example URL


        '''for index, row in df.iterrows():
                value = row[1]  # Access value at column index 1
                print(value)'''
        


        imprint = imprints_without_spaces
        color_input = color_output
        #shape_input = pills_shapes







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
                for color in color_input:
                    color = color.lower()  
                if color in color_to_number:
                    col_to_num = color_to_number[color]
                    return col_to_num
                else:
                    return None

        color_number = get_color_number(color_input)
    
        '''url = f"{root}imprint={imprint}"
        page = requests.get(url)
        print(url)
        if page.status_code == 200:  
            soup = BeautifulSoup(page.content, 'html.parser')

        drug_results = soup.find('a', class_='ddc-btn ddc-btn-small', href=True)  
        if drug_results:
            link_url = drug_results['href']
            print(f"Link URL: {drug_results['href']}")
        else:
            print("No matching anchor tag found.")


        from bs4 import BeautifulSoup
        import requests       
        base_url = "https://www.drugs.com"  
        full_url = base_url + link_url
        print(full_url)       
        response = requests.get(full_url)      
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser') 
            text_paragraphs1 = soup.find_all('h3', class_= "ddc-mgt-0")
            text_paragraphs2 = soup.find_all('dl', class_= "pid-list")
            text_paragraphs = text_paragraphs1 + text_paragraphs2
            drug_info = ""
            for paragraph in text_paragraphs:
                drug_info += paragraph.text.strip() + "\n"
                return drug_info.strip() 
                
                
        else:
            print(f"Error: Failed to retrieve linked page. Status code: {response.status_code}")

        return drug_info.strip()'''

        import requests
        from bs4 import BeautifulSoup

        def scrape_drug_information(imprint):
    # Construct the URL to search for the imprint
            root = "https://www.drugs.com/imprints_search.php?"
            url = f"{root}imprint={imprint}"

    # Send a GET request to the URL
            page = requests.get(url)
            print(url)
    
            if page.status_code == 200:
                soup = BeautifulSoup(page.content, 'html.parser')

        # Find the link to the specific drug page
                drug_results = soup.find('a', class_='ddc-btn ddc-btn-small', href=True)  
                if drug_results:
                    link_url = drug_results['href']
                    print(f"Link URL: {link_url}")
            
            # Construct the full URL for the drug page
                    base_url = "https://www.drugs.com"
                    full_url = base_url + link_url
                    print(f"Full URL: {full_url}")
            
            # Send a GET request to the drug page
                    response = requests.get(full_url)
            
                    if response.status_code == 200:
                # Parse the content of the linked webpage
                        soup = BeautifulSoup(response.content, 'html.parser') 
                
                # Find all elements with a specific class
                        drug_info_elements = soup.find_all('div', class_='drug-info')
                
                # Extract text from elements
                        drug_info = ""
                        for element in drug_info_elements:
                            drug_info += element.text.strip() + "\n"
                
                        return drug_info.strip()  # Remove trailing and leading whitespaces
                        
                    else:
                        print(f"Error: Failed to retrieve linked page. Status code: {response.status_code}")
                else:
                    print("No matching anchor tag found.")
            else:
                print("Error: Failed to retrieve search page.")

            return None

        drug_info = scrape_drug_information(imprint)
        if drug_info:
            print("Drug Information:")
            print(drug_info)


        '''import requests
        from bs4 import BeautifulSoup

        def scrape_drug_information(imprint):
    
            root = "https://www.drugs.com/imprints_search.php?"
            url = f"{root}imprint={imprint}"

    
            page = requests.get(url)
            print(url)
    
            if page.status_code == 200:
                soup = BeautifulSoup(page.content, 'html.parser')

        
                drug_results = soup.find('a', class_='ddc-btn ddc-btn-small', href=True)  
                if drug_results:
                    link_url = drug_results['href']
                    print(f"Link URL: {link_url}")
            
            
                    base_url = "https://www.drugs.com"
                    full_url = base_url + link_url
                    print(f"Full URL: {full_url}")
                    
            
                    response = requests.get(full_url)
            
                    if response.status_code == 200:
                
                        soup = BeautifulSoup(response.content, 'html.parser') 
                
                
                        text_paragraphs1 = soup.find_all('h3', class_="ddc-mgt-0")
                        text_paragraphs2 = soup.find_all('dl', class_="pid-list")
                        text_paragraphs = text_paragraphs1 + text_paragraphs2
                        drug_info = ""
                        for paragraph in text_paragraphs:
                            drug_info += paragraph.text.strip() + "\n"
                        return drug_info.strip()  
                
                    else:
                        print(f"Error: Failed to retrieve linked page. Status code: {response.status_code}")
                else:
                    print("No matching anchor tag found.")
            else:
                print("Error: Failed to retrieve search page.")

            return None

        
        imprint = imprints_without_spaces

        
        drug_info = scrape_drug_information(imprint)
        if drug_info:
            print("Drug Information:")
            print(drug_info)'''


demo = gr.Interface(fn=image_classifier,inputs="image",outputs="textbox",title="Drug Pill Identifier")
demo.launch(share=True)