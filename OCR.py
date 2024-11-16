from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler  
import cv2 as cv
import joblib
import matplotlib.pyplot as plt
  
LIST= {0: "0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B",
        12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H", 18:"I", 19:"J", 20:"K", 21:"L", 22:"M",
        23:"N", 24:"O", 25:"P", 26:"Q", 27:"R", 28:"S", 29:"T", 30:"U", 31:"V", 32:"W", 33:"X",
        34:"Y", 35:"Z"}

class OCRImplement:
  def __init__(self):
    self.W1= np.random.randn(35, 64)* np.sqrt(2./35)
    self.W2= np.random.randn(64, 256)* np.sqrt(2./64)
    self.W3= np.random.randn(256, 1024)* np.sqrt(2./256)
    self.W4= np.random.randn(1024, 1024)* np.sqrt(2./1024)
    self.W5= np.random.randn(1024, 1024)* np.sqrt(2./1024)
    self.W6= np.random.randn(1024, 256)* np.sqrt(2./1024)
    self.W7= np.random.randn(256, 64)* np.sqrt(2./256)
    self.W8= np.random.randn(64, 36)* np.sqrt(2./64)
  
    self.b1= np.zeros((1, 64))
    self.b2= np.zeros((1, 256))
    self.b3= np.zeros((1, 1024))
    self.b4= np.zeros((1, 1024))
    self.b5= np.zeros((1, 1024))
    self.b6= np.zeros((1, 256))
    self.b7= np.zeros((1, 64))
    self.b8= np.zeros((1, 36))
    
  def softMax(self, X):
    result= X - np.max(X, axis= 1, keepdims= 1)
    result= np.exp(result)
    result= result / np.sum(result, axis= 1, keepdims= 1)
    return result
  
  def lossFunction(self, yTrue):
    lossValue= np.sum(yTrue*self.A8, axis= 1, keepdims= True)
    result= -np.sum(np.log(lossValue))/yTrue.shape[0]
    return result
  
  def sigmoid(self, X):
    result= np.exp(-X)
    return 1/ (1+ result)

  def sigmoidDer(self, X):
    return self.sigmoid(X)*(1- self.sigmoid(X))
  
  def ReLU(self, Z):
    return np.maximum(Z, 0)

  def ReLUDer(self, Z):
    return Z > 0
  
  def feedForward(self, X):
    self.A0= X

    self.Z1= self.A0.dot(self.W1)+ self.b1
    self.A1= self.ReLU(self.Z1)
    
    self.Z2= self.A1.dot(self.W2)+ self.b2
    self.A2= self.ReLU(self.Z2)
    
    self.Z3= self.A2.dot(self.W3)+ self.b3
    self.A3= self.ReLU(self.Z3)
    
    self.Z4= self.A3.dot(self.W4)+ self.b4
    self.A4= self.ReLU(self.Z4)

    self.Z5= self.A4.dot(self.W5)+ self.b5
    self.A5= self.ReLU(self.Z5)

    self.Z6= self.A5.dot(self.W6)+ self.b6
    self.A6= self.ReLU(self.Z6)
    
    self.Z7= self.A6.dot(self.W7)+ self.b7
    self.A7= self.ReLU(self.Z7)
    
    self.Z8= self.A7.dot(self.W8)+ self.b8
    self.A8= self.softMax(self.Z8)
    
  def backPropagation(self, y, miniBatch):
    E8= (self.A8- y)/ miniBatch
    self.dW8= self.A7.T.dot(E8)
    self.db8= np.sum(E8, axis= 0, keepdims= True)
    
    E7= E8.dot(self.W8.T)
    E7= E7*self.ReLUDer(self.Z7)
    self.dW7= self.A6.T.dot(E7)
    self.db7= np.sum(E7, axis= 0, keepdims= True)
    
    E6= E7.dot(self.W7.T)
    E6= E6*self.ReLUDer(self.Z6)
    self.dW6= self.A5.T.dot(E6)
    self.db6= np.sum(E6, axis= 0, keepdims= True)
    
    E5= E6.dot(self.W6.T)
    E5= E5*self.ReLUDer(self.Z5)
    self.dW5= self.A4.T.dot(E5)
    self.db5= np.sum(E5, axis= 0, keepdims= True)
    
    E4= E5.dot(self.W5.T)
    E4= E4*self.ReLUDer(self.Z4)
    self.dW4= self.A3.T.dot(E4)
    self.db4= np.sum(E4, axis= 0, keepdims= True)
    
    E3= E4.dot(self.W4.T)
    E3= E3*self.ReLUDer(self.Z3)
    self.dW3= self.A2.T.dot(E3)
    self.db3= np.sum(E3, axis= 0, keepdims= True)
    
    E2= E3.dot(self.W3.T)
    E2= E2*self.ReLUDer(self.Z2)
    self.dW2= self.A1.T.dot(E2)
    self.db2= np.sum(E2, axis= 0, keepdims= True)
    
    E1= E2.dot(self.W2.T)
    E1= E1*self.ReLUDer(self.Z1)
    self.dW1= self.A0.T.dot(E1)
    self.db1= np.sum(E1, axis= 0, keepdims= True)
    
  def updateWeightBias(self, lr):
    self.W1-= lr*self.dW1
    self.W2-= lr*self.dW2
    self.W3-= lr*self.dW3
    self.W4-= lr*self.dW4
    self.W5-= lr*self.dW5
    self.W6-= lr*self.dW6
    self.W7-= lr*self.dW7
    self.W8-= lr*self.dW8
    
    self.b1-= lr*self.db1
    self.b2-= lr*self.db2
    self.b3-= lr*self.db3
    self.b4-= lr*self.db4
    self.b5-= lr*self.db5
    self.b6-= lr*self.db6
    self.b7-= lr*self.db7
    self.b8-= lr*self.db8
  
  def convertOneHotCoding(self, y):
    max= np.max(y)
    y_oneHot= np.zeros((y.shape[0], max+ 1))
    for i in range (y_oneHot.shape[0]):
        y_oneHot[i, max - y[i, 0]]= 1
    return y_oneHot 
  
  def fitGradient(self, X, y, lr, miniBatch):
    self.lda= LinearDiscriminantAnalysis()
    self.lda.fit(X, np.ravel(y))
    X= self.lda.transform(X)
    
    self.scaler= MinMaxScaler()
    self.scaler.fit(X)
    X= self.scaler.transform(X)
    y= self.convertOneHotCoding(y)
    pairs= X.shape[0]// miniBatch
    flagFirstTime= True
    numberOfEpoch= 0
    self.ArrayNumberEpoch= []
    self.ArrayLossValue= []
    previousLossValue= 0
    while True:
      list_id= np.random.permutation(pairs*miniBatch).reshape(pairs, miniBatch)
      for i in range (pairs):
        X_sub= X[list_id[i, :], :]
        y_sub= y[list_id[i, :], :]
        self.feedForward(X_sub)
        previousLossValue= self.lossFunction(y_sub)
        if flagFirstTime:
          self.ArrayNumberEpoch.append(numberOfEpoch)
          self.ArrayLossValue.append(previousLossValue)
          flagFirstTime= False
        if previousLossValue  < 0.2:
          return;
        else:
          print(previousLossValue)
        self.backPropagation(y_sub, miniBatch)
        self.updateWeightBias(lr)
      numberOfEpoch+= 1
      self.ArrayNumberEpoch.append(numberOfEpoch)
      self.ArrayLossValue.append(previousLossValue)
  
  def predict(self, X):
    X= self.lda.transform(X)
    X= self.scaler.transform(X)
    self.feedForward(X)
    maxArray= np.argmax(self.A8, axis= 1, keepdims= 1)
    return (len(LIST)-1 - maxArray)
    
    
  def accuracyFunction(self, y_predict, y):
    total= np.sum(np.all(y_predict== y, axis= 1)).astype((float))
    return (total/ y_predict.shape[0])* 100
    
def resize(path):
  pathBaseFolder= path
  for i in range(0, len(LIST)):
    pathSubFolder= pathBaseFolder+ LIST[i]
    item= os.listdir(pathSubFolder)
    for subItem in item:
      imagePath= pathSubFolder+ "/"+ subItem
      image= cv.imread(imagePath)
      image= cv.resize(image, (25, 35))
      cv.imwrite(imagePath, image)

def drawLossValueEpoch(X, y):
  plt.figure(figsize=(8, 6))
  # plt.scatter(x, y, color='blue')  # Plot the scatter points
  plt.plot(X, y, color='red', linewidth=1 )  # Plot the regression line
  plt.xlabel('Epoches')
  plt.ylabel('Loss Value')
  plt.title('Relationship between X and Y with Regression Line')
  plt.show()
    
def ConvertImageCharacter(image):
  HSV= cv.cvtColor(image, cv.COLOR_BGR2HSV)
  H, S, V= cv.split(HSV)
  
  imageGray= V
  imageGray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  kernel= cv.getStructuringElement(cv.MORPH_RECT, (3, 3))


  imageTopHat= cv.morphologyEx(imageGray, cv.MORPH_TOPHAT, kernel, iterations= 7)
  imageBlackHat= cv.morphologyEx(imageGray, cv.MORPH_BLACKHAT, kernel, iterations= 7)

  imagePlusTopHat= cv.add(imageGray, imageTopHat)
  imagePlusTopHatMinusBlackHat= cv.subtract(imagePlusTopHat, imageBlackHat)
  
  imageGaussNoise= cv.GaussianBlur(imagePlusTopHatMinusBlackHat, (3, 3), 7)

  # imageAdaptiveThreshold= cv.adaptiveThreshold(imageGaussNoise, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 29, 9)
  _, imageThreshold= cv.threshold(imageGaussNoise, 100, 255, cv.THRESH_BINARY)

  return imageThreshold

    
def writeCSV(path, fileName):
  fileCSV= open(fileName, 'a')
  for i in range(0, len(LIST)):
    pathFolder= path+ LIST[i]
    item= os.listdir(pathFolder)
    for imageName in item:
      pathImage= pathFolder + "/"+ imageName
      image= cv.imread(pathImage)
      # image= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
      image= ConvertImageCharacter(image)
      vector= image.flatten()
      for order in vector:
        fileCSV.write(str(order))
        fileCSV.write(",")
      fileCSV.write(str(i))
      fileCSV.write('\n')
  fileCSV.close()



if __name__== "__main__":
  # pathOfDataTrain= "DatasetOCRImplement/TRAINING_SET/"
  # pathOfDataTest= "DatasetOCRImplement/TEST_SET/"
  
  # fileCSVTrain= "datasetTrain.csv"
  # fileCSVTest= "datasetTest.csv"
  
  # resize(pathOfDataTrain)
  # resize(pathOfDataTest)
  
  # writeCSV(pathOfDataTrain, fileCSVTrain)
  # writeCSV(pathOfDataTest, fileCSVTest)
  
  
  frameTrain= pd.read_csv("datasetTrain.csv")
  XTrain= frameTrain.iloc[:, 0: frameTrain.shape[1]-1].values
  yTrain= frameTrain.iloc[:, frameTrain.shape[1]-1:].values
  
  frameTest= pd.read_csv("datasetTest.csv")
  XTest= frameTest.iloc[:, 0: frameTest.shape[1]-1].values
  yTest= frameTest.iloc[:, frameTest.shape[1]-1:].values
  
  
  objectANN= OCRImplement()
  
  objectANN.fitGradient(XTrain, yTrain, lr= 0.001, miniBatch= 500)
  
  drawLossValueEpoch(objectANN.ArrayNumberEpoch, objectANN.ArrayLossValue)
  
  
  yPredict= objectANN.predict(XTest)
  
  print("Accuracy: ", objectANN.accuracyFunction(yPredict, yTest), "%")
  
  
  joblib.dump(objectANN, "trainModel.joblib")