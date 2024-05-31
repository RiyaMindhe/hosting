import pandas as pd
import cv2
import easyocr
img = cv2.imread(r"C:\Users\HP\Desktop\Final Year Project\pill_recognition\pill_pictures\1\1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise=cv2.medianBlur(gray,3)
thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
reader = easyocr.Reader(["en"], verbose=False)
result = reader.readtext(img,paragraph="False")
df=pd.DataFrame(result)
print(df[1])