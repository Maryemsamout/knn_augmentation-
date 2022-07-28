import math
import numpy as np
import os
import cv2
import math
def rotation(image,phi):
    size=image.shape
    longeur=size[0]
    largeur=size[1]
    newImage=np.zeros((longeur,largeur))
    
    for i in range(longeur):
        for j in range(largeur):
            try:
                newImage[i,j]=image[round(math.cos(phi)*i)-round(math.sin(phi)*j),round(math.sin(phi)*i)+round(math.cos(phi)*j)]
            except:
                print('out of rotation' +str((i,j)))
    
    return newImage

def translation(image,x0,y0):
    size=image.shape
    longeur=size[0]
    largeur=size[1]
    newImage=np.zeros((longeur,largeur))
    
    for i in range(longeur):
        for j in range(largeur):
            try:
                newImage[i+x0,j+y0]=image[i,j]
            except:
                print('out of translation' +str((i,j)))
    return newImage

def convertToRadius(phi):
    return (phi/360)*2*math.pi

def ImageToClasse(imgName):
    k=0
    ch=''
    while(imgName[k]!='.'):
        ch=ch+imgName[k]
        k=k+1
        
    return ch

def centreRotation(image,phi):
    size=image.shape
    longeur=size[0]
    largeur=size[1]
    integralxfx=0
    integralf=0
    integralyfx=0
    
    for i in range(longeur):
        for j in range(largeur):
            integralf=integralf+image[i,j]
            integralxfx=integralxfx+image[i,j]*i
            integralyfx=integralyfx+image[i,j]*j
    
    xg=integralxfx/integralf
    yg=integralyfx/integralf
    
    xg=round(xg)
    yg=round(yg)
    
    newImage=np.zeros((longeur,largeur))
    
    for i in range(longeur):
        for j in range(largeur):
            try:
                newX=round((i-xg)*math.cos(phi)-(j-yg)*math.sin(phi))+xg
                newY=round((i-xg)*math.sin(phi)+(j-yg)*math.cos(phi))+yg
                newImage[newX,newY]=image[i,j]
                
            except:
                print(i,j)
    return newImage

dataImage=[]
dir=r'C:\Users\asus\Desktop\knn_mpeg\PNG_data2'
dir1=r'C:\Users\asus\Desktop\knn_mpeg\dataAug'
k=0
for img in os.listdir(dir):
    k=k+1
    if k>=0:
        newDir=dir+"/"+img
        image=cv2.imread(newDir,2)
        image=cv2.resize(image,(250,250))
        dataImage.append(image)
        k=k+1
        for i in range(2,3):
           phi=convertToRadius(2*math.pi/i)
           try:
               rotImage=centreRotation(image,phi)
           
               rotImage1=centreRotation(image, phi)
           except:
               print(phi)
           
            
           cv2.imwrite(dir1+'/'+ImageToClasse(img)+'-x4t'+str(k)+str(i)+'.png',rotImage)
           
           cv2.imwrite(dir1+'/'+ImageToClasse(img)+'-x4t1'+str(k)+str(i)+'.png',rotImage1) 
    else:
         print(k)


