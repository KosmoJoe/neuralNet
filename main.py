##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, Main.py, is part of NeuralNet and is for playing 
##  $Id: Main.py

from neuralNet.Net import Net
from data.Samples import Samples
from neuralNet.CostFunctions import Entropy
import time
import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
from data.DisplayData import DisplayData

#### DATA LOADING 
np.set_printoptions(threshold=np.nan)
matfile = sio.loadmat('data/data1.mat')

#### Making of Training Sample
numExamples = 500#matfile['y'].shape[0]
numFeatures = matfile['X'].shape[1]
numClasses = 10
X = matfile['X'][range(numExamples),:]
y = matfile['y'][range(numExamples)]
newSample = Samples([numFeatures,numExamples, numClasses])
newSample.xdata = X.transpose()
newSample.y = y
newSample.label = np.arange(10)+1

#### Set-up of Neural Net
myNet = Net(2,[25, numClasses])    ## Two Layer, 25 Hidden, 10 output nodes
myNet.trainSet = newSample
myNet.setParams([0.5,0.001,3000])  ## Rate, Regularization, Epochs
myNet.setFunctions([lambda x: (np.tanh(x)+1)/2, lambda x: (1 - np.tanh(x)**2)/2], Entropy())
myNet.initializeNet()

#### Train Neural Net

start = time.time()
myNet.train()
end = time.time()
print("Time for Training computation:" + str(end-start))
print("Classification Error: " + str(myNet.classificationAccuray()))


#### Plot the results
plt.plot(myNet._cost,label='cost', linewidth=2)
plt.plot(myNet._acc,label='acc', linewidth=2)
plt.show()
DisplayData(myNet._layer[0].weights,20)



