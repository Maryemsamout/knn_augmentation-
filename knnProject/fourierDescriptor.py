#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation,Conv1D, Flatten, BatchNormalization
#from tensorflow.keras.utils import to_categorical
import Pavlids as pd
import observerImages
import numpy as np
import reparemetrageEuclidien as rd
import cv2
#import random
import os 
import matplotlib.pyplot as plt
import classifier

def affiche(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    
def show(x,y):
    plt.scatter(x,y)
    
def norme(xi,yi) :
    zi = [(i * 1j) + h for i,h  in zip(yi,xi)]
    fourier = np.fft.fft(zi)
    I =fourier/fourier[1]
    return abs(I)

def FourierDescriptor(observerImages):
    x1=[]
    y1=[]
    for img in os.listdir(observerImages.getPath()):
        try:
            image=cv2.imread(observerImages.getPath()+'/'+img,0)
            
            image=cv2.Laplacian(image,cv2.CV_8U,ksize=1)
            # affiche(image)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            # image = cv2.dilate(image, kernel)
            contour_complex=pd.pavlidis(image)
            x,y=rd.Reparametrage_euclidien(contour_complex.imag,contour_complex.real,100)
            classe=observerImages.getClasse(img)
            I=norme(x,y)
            x1.append(I)
            y1.append(classe)
        except:
            print("error in "+img)
    return x1,y1


path=r'C:\Users\asus\Desktop\knn_mpeg\PNG_data2'
observerImages=observerImages.observerImages(path)
x,y=FourierDescriptor(observerImages)
cl=classifier.classifier(x,y)
scoreKnn=cl.knn()



            
            
    
    