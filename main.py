
import cv2 as cv
# from lib_detection import load_model, detect_lp, im2single
import math
import numpy as np
import time
from PIL import Image
from OCR import OCRImplement
import joblib


  
LIST= {0: "0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B",
        12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H", 18:"I", 19:"J", 20:"K", 21:"L", 22:"M",
        23:"N", 24:"O", 25:"P", 26:"Q", 27:"R", 28:"S", 29:"T", 30:"U", 31:"V", 32:"W", 33:"X",
        34:"Y", 35:"Z"}

def convertImageInverse(image):
  a= -1
  b = 255
  rows= image.shape[0]
  columns= image.shape[1]
  
  for i in range(rows):
    for j in range(columns):
      image[i][j]= a*image[i][j] + b
  return image

def ConvertImageCharacter(image):
    imageGrayOriginal= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel= cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    
    imageInverse = convertImageInverse(imageGrayOriginal)
    
    imageInverseClosing= cv.morphologyEx(imageInverse, cv.MORPH_CLOSE, kernel)
    
    imageInverseClosingOpening = cv.morphologyEx(imageInverseClosing, cv.MORPH_OPEN, kernel)
    
    imageGaussNoise= cv.GaussianBlur(imageInverseClosingOpening, (3, 3), 5)

    # imageAdaptiveThreshold= cv.adaptiveThreshold(imageGaussNoise, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 29, 9)
    _, imageThreshold= cv.threshold(imageGaussNoise, 130, 255, cv.THRESH_BINARY)
    
    return convertImageInverse(imageThreshold)

def main():
    fileName = "result.txt"
    model= joblib.load("trainModel.joblib")
    pathImage= "test.jpg"

    image= cv.imread(pathImage)
 
    imageThreshold= ConvertImageCharacter(image)
    
    cv.imshow("image", imageThreshold)
    cv.waitKey(0)
    contour, _= cv.findContours(imageThreshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    ArrayArea= []
    for i in contour:
        x, y, w, h= cv.boundingRect(i)
        if  w*h > 100 and 0.8 <= h/w < 5:
            ArrayArea.append(w*h)
            
    SortedArrayArea= sorted(ArrayArea)
    
    Contour8= []
    XCoordinate= []
    for i in contour:
        x, y, w, h= cv.boundingRect(i)
        if w*h in SortedArrayArea:
            XCoordinate.append(x)
            Contour8.append(i)
            
    XCoordinateSorted= sorted(XCoordinate)
            
    string= ""
    for i in XCoordinateSorted:
        index= XCoordinate.index(i)
        x, y, w, h= cv.boundingRect(Contour8[index])
        subImage= imageThreshold[y -1: y+ h+ 1, x-1 : x+ w+ 1]
        subImage= cv.resize(subImage, (25, 35))
        vector= subImage.flatten().reshape(1, -1)
        string+= LIST[model.predict(vector)[0][0]]
        image= cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    with open(fileName, "w") as file:
        file.write(string)
        file.close()
    cv.imshow("Rotation", image)
    cv.waitKey(0)

if __name__== "__main__":
    main()
   
    