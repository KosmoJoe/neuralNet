##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, Net.py, is part of NeuralNet and contains the class 
##  which implements a network of neurons including all its properties
##  $Id: Net.py

import numpy as np
from neuralNet.Neuron import Neuron
from neuralNet.Layer import Layer
from data.Samples import Samples
from neuralNet.CostFunctions import Entropy,LeastSquare


#No underscore: it's a public variable.
#One underscore: it's a protected variable.
#Two underscores: it's a private variable. 

########################################################################
# Begin class Net
######
class Net(object):
    """
    Define a Neuron of a neuronal Network
    """

##### Class atrributes
    # type = "Neuron"
    __DEBUG_Net = False

    def __init__(self,numOfLayers, structure):
        """
        Initialize a net:
        Input: structure = [ numOfLayers, numHiddenNodes[] ]
        """
        self.__isInitialized = False                  # Flag to note if all parameter are set
        
        
        self.__numOfLayers = numOfLayers                       # Number of Layers in the Net (Input does not count)
        self.__numHiddenNodes = structure                    # Number of Hidden Neurons per Layer; len = numOfLayers-1
        self.__numFeatureMaps = []                    # Number of Features Maps per Layer; len = numOfLayers-1
        
        self._trainSet = None                         # Pointer to training set
        self._layer = []                              # Layer List

        self._costFunc = None                         # Pointer to cost function object
        self._cost = []                             # Cost array[decade]
        self._acc = []                              # Accuracy [decade] 
        self._decade = 0                              # Current decade of training
        self._totdecades = 100                        # Total number of decades

        self._rate = 0.1                   # Learning rate
        self._reg = 0.1                    # Regularization Parameter
        self._epsiWeight = 0.012            # Range for weights to be initialized
        self._activationFunction = None  # Pointer to activation function f(wt*x + b)
        self._derivativeActivationFunction = None  # Pointer to derivative of activation function f(wt*x + b)
        
###################
# Class properties
###################

    #### Constants
    @property
    def numOfLayers(self):
        return self.__numOfLayers
    @property
    def numHiddenNodes(self):
        return self.__numHiddenNodes    


    @property
    def trainSet(self):
        return self._trainSet
    @trainSet.setter
    def trainSet(self, val):
        if isinstance(val,Samples) :
            self._trainSet = val
            if Net.__DEBUG_Net: print("Sample Object has been set")
        else:
            self._trainSet = None

    def setParams(self,params):
        """
        Function to set learning parameters
        """
        self._rate = params[0]
        self._reg  = params[1]
        self._totdecades  = params[2]
    
    def setFunctions(self,activation,cost):
        """
        Function to set learning functions
        """
        self._activationFunction = activation[0]
        self._derivativeActivationFunction = activation[1]
        self._costFunc  = cost


    def initializeNet(self):
        """
        Create Layers and set their data
        """
        #if not self.__isInitialized:
        #    self._check_initialize()
            
        for k in range(self.__numOfLayers): # run over neurons
            
            
            if k==0:
                newLayer = Layer([self._trainSet.numExamples, self.__numHiddenNodes[k],self._trainSet.numFeatures,self._trainSet.numFeatures,1])
                newLayer.features = self.trainSet.xdata
                newLayer.weights = self.randInitWeights([self._trainSet.numFeatures,self.__numHiddenNodes[k]])
                newLayer.bias = self.randInitWeights([self.__numHiddenNodes[k],1])
            else:
                newLayer = Layer([self._trainSet.numExamples, self.__numHiddenNodes[k],self.__numHiddenNodes[k-1],self.__numHiddenNodes[k-1],1])
                newLayer.features = self._layer[-1].output[0,:]
                newLayer.weights = self.randInitWeights([self.__numHiddenNodes[k-1],self.__numHiddenNodes[k]])
                newLayer.bias = self.randInitWeights([self.__numHiddenNodes[k],1])
                self._layer[-1]._nextLayer = newLayer

            newLayer.setState(k,k+1!=self.__numOfLayers)
            newLayer._costFunc = self._costFunc
            newLayer.output = np.zeros((1,self.__numHiddenNodes[k],self._trainSet.numExamples))
            newLayer.error = np.zeros((1,self.__numHiddenNodes[k],self._trainSet.numExamples))
            newLayer.helpVector = np.zeros((self.__numHiddenNodes[k],self._trainSet.numExamples))
            newLayer.activationFunction = self._activationFunction
            newLayer.derivativeActivationFunction = self._derivativeActivationFunction
            newLayer.trainingSet = self._trainSet
            newLayer.setParams([self._rate,self._reg])
            newLayer.initializeLayer()
            self._layer.append(newLayer)

        self.__isInitialized = True 
    
    def feedforward(self):
        """
        Feedforward for whole network
        """
        if not self.__isInitialized:
            raise ValueError('Warning: Layer Neurons have not been initialized yet')
        for k in range(self.__numOfLayers): # run over neurons
            self._layer[k].feedforward(False)

    def backprop(self):
        """
        Backprop for whole network
        """
        if not self.__isInitialized:
            raise ValueError('Warning: Layer Neurons have not been initialized yet')
        for k in reversed(range(self.__numOfLayers)): # run over neurons
            self._layer[k].backprop()

    def update(self):
        """
        Update for whole network
        """
        for k in range(self.__numOfLayers): # run over neurons
            self._layer[k].updateWeights()

    

    def train(self):
        """
        Train of given train set
        """
        for dec in range(self._totdecades):
            self._decade = dec
            self.feedforward()
            self.backprop()
            self.update()
            self._cost.append(self.cost())
            self._acc.append(self.classificationAccuray()/100)

    def predict(self):
        """
        Derive prediction for set of samples
        """
        return self._trainSet.label[np.argmax(self._layer[-1].output,axis=1)]

    def classificationAccuray(self):
        """
        Derive prediction for set of samples
        """
        temp = (self._layer[-1].output[0,:] == np.max(self._layer[-1].output,axis=1)).astype(int)
        return np.sum((temp & self._trainSet.ydata).astype(int))/self._trainSet.numExamples*100


    def cost(self):
        """
        Derive cost function 
        """
        #cost = np.sum(np.abs(self._layer[-1].output[0,:]-self._trainSet.ydata)**2)
        cost = self._costFunc.cost(self._layer[-1].output[0,:], self._trainSet.ydata)
        for kk in range(self.__numOfLayers):
            cost+= self._reg*np.sum(np.dot(self._layer[kk].weights,self._layer[kk].weights.transpose()))/self._trainSet.numExamples
        #self._cost[self._decade] = cost
        return cost


    def randInitWeights(self,dims):
        """
        Function to initialize weights in a certain region
        """
        if len(dims)==1:
            return np.random.rand(dims[0])*self._epsiWeight*2 - self._epsiWeight
        elif len(dims) ==2 :
            return np.random.rand(dims[0],dims[1])*self._epsiWeight*2 - self._epsiWeight
        else:
            raise ValueError("Format not supported")


    def computeNumericGrad(self):
        """
        Derive the numerical gradient to check backprop algo
        """
        pass
        #epsilon  = 1e-4
        #for weight in self.weights:
######
# End class Net
########################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import scipy.io as sio
    from DisplayData import DisplayData
    np.set_printoptions(threshold=np.nan)
    matfile = sio.loadmat('data1.mat')
    numExamples = 5000#matfile['y'].shape[0]
    numFeatures = matfile['X'].shape[1]
    #print(matfile['X'].shape)
    numClasses = 10
    X = matfile['X'][range(numExamples),:]
    y = matfile['y'][range(numExamples)]
    #help_y = np.ones((1,numExamples))*np.arange(1,numClasses+1)[:, np.newaxis]  # Promote to vectors of ones 
    #help_y = (help_y == y)
    #print(help_y == y)
    #print(y)
    #print(X.shape)
    #print(y.shape)
    newSample = Samples([numFeatures,numExamples, numClasses])
    newSample.xdata = X.transpose()
    #selection = newSample.xdata.transpose().copy()
    #np.random.shuffle(selection)
    #selection = selection.transpose()
	
    newSample.y = y#(help_y==y.transpose()).astype(int)
    newSample.label = np.arange(10)+1
    #print(newSample.y[0:100])

    myNet = Net(2,[25, numClasses])
    myNet.trainSet = newSample
    myNet.setParams([0.5,0.001,3000])
    myNet.setFunctions([lambda x: (np.tanh(x)+1)/2, lambda x: (1 - np.tanh(x)**2)/2],Entropy())
    myNet.initializeNet()
    #myNet.feedforward()
    #print(myNet._layer[3]._nextLayer)
    #myNet.backprop()
    #print(myNet.cost())
    #print(myNet._layer[3])
    #print(np.sum(myNet._layer[2].weights))
    #print(myNet.cost())
    myNet.train()
    #print(myNet.predict().shape)
    print(myNet.classificationAccuray())
    #print(myNet._layer[-1].output[0,:,300])
    #print(myNet._trainSet.ydata[:,300])
    #print(myNet.cost())
    #print(np.sum(myNet._layer[2].weights))
    #print(myNet._layer[2].output[0,:,1])
    plt.plot(myNet._cost,label='cost', linewidth=2)
    plt.plot(myNet._acc,label='acc', linewidth=2)
    plt.show()
    DisplayData(myNet._layer[0].weights,20)
    #DisplayData(myNet._layer[0].features[:,0:25],20)
    