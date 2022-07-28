
import Pavlids as pd
import observerImages
import numpy as np
import reparemetrageEuclidien as rd
import cv2
import random
import os 
import classifier
def crimmins_stable(xi,yi,n0,n1,p,q):
    
    zi = [(i * 1j) + h for i,h  in zip(yi,xi)]
    fourier = np.fft.fft(zi)
    I = list()
    for i in range(len(zi)) :
        In = (fourier[i]**(n0-n1)) * (fourier[n0-1]**(n1-i)) * (fourier[n1-1]**(i-n0))
        Jn = (fourier[n0-1]**(n1-i-p)) * (fourier[n1-1]**(i-n0-q))
        
        I.append(In/Jn)
    #f=[]
    #for i in I:
    #f.append(abs(i))
    return I

def GhorbelDescriptor(observerImages):
    x1=[]
    y1=[]
    for img in os.listdir(observerImages.getPath()):
        try:
            image=cv2.imread(observerImages.getPath()+'/'+img,0)
            # image=cv2.GaussianBlur(image,(3,3),0)
            # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image=cv2.Laplacian(image,cv2.CV_8U,ksize=1)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            # image = cv2.dilate(image, kernel)
            contour_complex=pd.pavlidis(image)
            a,b=pd.getXY(contour_complex)
            x,y=rd.Reparametrage_euclidien(a,b,80)
            classe=observerImages.getClasse(img)
            
            I=crimmins_stable(x,y,3,2,1/2,1/2)
            x1.append(I)
            y1.append(classe)
            print("succes read "+img)
            
            # print(data)
        
        except:
             print(img)
    return x1,y1


path=r'C:\Users\asus\Desktop\knn_mpeg\PNG_data2'
observerImages=observerImages.observerImages(path)
x,y=GhorbelDescriptor(observerImages)
x1=[]
for i in x:
    
    x1.append(np.absolute(i))
    
cl=classifier.classifier(x1,y)
scoreKnn=cl.knn()


            
    
    

