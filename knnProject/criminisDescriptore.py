
import Pavlids as pd
import observerImages
import numpy as np
import reparemetrageEuclidien as rd
import cv2
#import random
import os 
import classifier
def crimmins(xi,yi,n0,n1):
    zi = [(i * 1j) + h for i,h  in zip(yi,xi)]
    tfourier = np.fft.fft(zi)
    fourier = np.fft.fftshift(tfourier)
    I = list()
    for i in range(-len(zi)//2,len(zi)//2) :
        In = (fourier[i]**(n0-n1)) * (fourier[n0-1]**(n1-i)) * (fourier[n1-1]**(i-n0))
        I.append(In)
    return I

def criminisDescriptor(observerImages):
#    data=[]
    x1=[]
    y1=[]
    for img in os.listdir(observerImages.getPath()):
        try:
            image=cv2.imread(observerImages.getPath()+'/'+img,0)
            contour_complex=pd.pavlidis(image)
            x,y=rd.Reparametrage_euclidien(contour_complex.real,contour_complex.imag,100)
            classe=observerImages.getClasse(img)
            I=crimmins(x,y,3,2)
            x1.append(I)
            y1.append(classe)
            
            
            print("succes read "+img)
            # print(data)
        except:
            print("error in "+img)
    return x1,y1

path=r'C:\Users\asus\Desktop\knn_mpeg\PNG_data2'
observerImages=observerImages.observerImages(path)
x,y=criminisDescriptor(observerImages)
x1=[]
for i in x:
    
    x1.append(np.absolute(i))
cl=classifier.classifier(x1,y)
scoreKnn=cl.knn()


            
            
    
    