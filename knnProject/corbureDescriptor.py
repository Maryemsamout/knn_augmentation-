from scipy.interpolate import *
import Pavlids as pd
import observerImages
import numpy as np
import reparemetrageEuclidien as rd
import cv2
import random
import os

import classifier
def courbure(x,y) :
    
    nx=len(x)-1
    t = list(np.arange(0,1,1/nx))
    if(len(t) != len(x)) :
        t.append(1.)
    
    x_t = splrep(t,x, s=0)   
    y_t = splrep(t,y, s=0)
    
    x_1 = splev(t,x_t, der=1)  # x'(t)
    x_2 = splev(t,x_t, der=2)  # x"(t)
    
    y_1 = splev(t,y_t, der=1)  # y'(t)
    y_2 = splev(t,y_t, der=2)  # y"(t)
    
    
    k = list () 
    for i in range(len(x)) :
        a = (x_1[i] * y_2[i]) - (y_1[i] * x_2[i])
        b = ((x_1[i]**2) + (y_1[i]**2))**1.5
        res = a/b
        k.append(abs(res))
    
    return k

def corbureDescriptor(observerImages):
    data=[]
    x1=[]
    y1=[]
    for img in os.listdir(observerImages.getPath()):
        try:
            image=cv2.imread(observerImages.getPath()+'/'+img,0)
            contour_complex=pd.pavlidis(image)
            x,y=rd.Reparametrage_euclidien(contour_complex.real,contour_complex.imag,20)
            classe=observerImages.getClasse(img)
            I=courbure(x,y)
            x1.append(I)
            y1.append(classe)
            
            print('read succes in '+img)
            # print(data)
        except:
            print("error in "+img)
    return x1,y1

path=r'C:\Users\asus\Desktop\knn_mpeg\PNG_data2'
observerImages=observerImages.observerImages(path)
x,y=corbureDescriptor(observerImages)
cl=classifier.classifier(x,y)
scoreKnn=cl.knn()


            
            
    
    

