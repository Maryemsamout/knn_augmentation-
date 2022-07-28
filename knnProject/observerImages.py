import os
import numpy as np
class observerImages:
    
    def __init__(self,path):
        self.path=path
        
    def getPath(self):
        return self.path
    
    def imageToCategorie(self,imageName):
        nameCategorie=''
        k=0
        while (imageName[k] !='-') :
            nameCategorie+=imageName[k]
            k+=1
        return nameCategorie
    
    def getClasse(self,imageName):
        classeName=self.imageToCategorie(imageName)
        return self.makeCategorieList().index(classeName)
        
    def makeCategorieList(self):
        categories=[]
        for img in os.listdir(self.path):
            categorie=self.imageToCategorie(img)
            if categorie not in categories:
                categories.append(categorie)
        return categories
    
    def FromDataToArray(self,data):
        nbFeatures=len(data[0][0])
        arrayData=np.zeros((nbFeatures+1,len(data)))
        count=0
        for imageData in data:
            arrayData[nbFeatures,count]=imageData[1]
            for i in range(nbFeatures):
                arrayData[i,count]=imageData[0][i]
            count+=1
        return arrayData
        
    