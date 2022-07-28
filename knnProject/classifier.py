from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB as bayes
from sklearn.utils import shuffle
import numpy as np
class classifier:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    
    def deviseData(self):
        longeur=len(self.x)
        longeurtrain=int(longeur*80/100)
        self.x,self.y=shuffle(self.x,self.y)
        xtrain=self.x[:longeurtrain]
        ytrain=self.y[:longeurtrain]
        xtest=self.x[longeurtrain:longeur]
        ytest=self.y[longeurtrain:longeur]
        return xtrain,ytrain,xtest,ytest
    
    def knn(self,k=1):
        neight=KNeighborsClassifier(n_neighbors=k)
        xtrain,ytrain,xtest,ytest=self.deviseData()
        neight.fit(xtrain,ytrain)
        score=neight.score(xtest,ytest)
        print("score knn= "+str(score))
        return score
    
    
        
    
    

    
    
    

    
        
        
        