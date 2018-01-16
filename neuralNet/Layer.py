##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, Layer.py, is part of NeuralNet and contains the class 
##  which implements a single network layer
##  $Id: Layer.py

import numpy as np
from neuralNet.Neuron import Neuron


#No underscore: it's a public variable.
#One underscore: it's a protected variable.
#Two underscores: it's a private variable. 

########################################################################
# Begin class Layer
######
class Layer(Neuron):
    """
    Define a Layer of a neuronal Network
    """

##### Class atrributes
    # type = "Neuron"
    __DEBUG_Layer = False
    __perNeuron = False

    def __init__(self,structure):
        """
        Initialize a layer:
        Input: structure = [numBatch, numNeuron, numIn, numWeights, numOut]
        """
        Neuron.__init__(self,[structure[0], structure[2], structure[3], structure[1]*structure[4]])
        #self.__isInitialized = False                  # Flag to note if all parameter are set
        self.__hasDataSet = False                     # Flag to test if data pointer are set
        self.__testGradient = True
        
        self.__numOutPerNeuron = structure[4]         # Number of output values (num of out neurons)
        #self.__numWeightPerNeuron = structure[3]      # Number of output values
        self.__numNeuron = structure[1]              # Number of Neurons of this Layer
        self.__numOutFeatures = 1                     # Number of output features / dimension of the output
        
        
        #self.__numNodes = 0                          # Number of Neurons in Layer
        #self.__numFeatureMaps = 0                    # Number of Features Maps 

        self._neurons = []               # List of Neurons ... becomes list of lists for multiple feature maps
        self._nextLayer = None           # Pointer to next layer in network  

        self._error = None               # Pointer to error delta[] element (self.numNeuron)x(self.numBatch)
        self.trainingSet = None

        self._costFunc = None            # Pointer to cost function object

        self._gradCostWeights = None         # Pointer to current gradient of cost for weights
        self._gradCostBias = None            # Pointer to current gradient of cost for Bias
        
        #self._features = None            # Pointer to features x[featureIndex,sampleIndex] element (self.__numIn)x(self.__numBatch)
        #self._weights = None             # Pointer to weigths  w[featureIndex,outputIndex] element (self.__numWeight)x(self.__numNeuron*self.__numOut)
        #self._output = None              # Pointer to output   a[outputIndex,sampleIndex]  element (self.__numOutFeatures)x(self.__numNeuron*self.__numOut)x(self.__numBatch)


###################
# Class properties
###################

    #### FEATURES
    @property
    def neurons(self):
        return self._neurons

    @Neuron.weights.setter
    def weights(self, val):
        if val.shape == (self.numWeight, self.numOut):
            self._weights = val
            if Layer.__DEBUG_Layer: print("Weights have been set")
        else:
            self._weights = None

    @Neuron.output.setter
    def output(self, val):
        dims = (self.__numOutFeatures, self.numOut, self.numBatch)
        if val.shape == dims:
            self._output = val
            if Layer.__DEBUG_Layer: print("Output is set")
        else:
            self._output = None
    
    @Neuron.bias.setter
    def bias(self, val):
        dims = (self.__numNeuron,1)
        if val.shape == dims:
            self._bias = val
            if Layer.__DEBUG_Layer: print("Bias is set")
            self.hasBias = True
        else:
            self._bias = None

    #### Error
    @property
    def error(self):
        return self._error
        
    @error.setter       ## Error setter to use external memory
    def error(self, val):
        dims = (self.__numOutFeatures, self.numOut, self.numBatch)
        if val.shape == dims:
            self._error = val
            if Layer.__DEBUG_Layer: print("Error is set")
        else:
            self._error = None

###################
# Class functions
###################

    def _check_initialize(self):
        """
        Check if all needed input was supplied
        """
        Neuron._check_initialize(self)

    def setState(self,layerNum,isHidden):
        super().setState(layerNum,0,0,isHidden)

    def initializeLayer(self):
        """
        Create Neurons and set their data
        """
        if not self.isInitialized:
            self._check_initialize()
            
        for m in range(1):  #run over feature maps
            lst = []
            for k in range(self.__numNeuron): # run over neurons
                newNeuron = Neuron([self.numBatch,self.numIn,self.numWeight,self.__numOutPerNeuron])
                newNeuron.weights = self.weights[:,k*self.__numOutPerNeuron:(k+1)*self.__numOutPerNeuron]
                newNeuron.features = self.features[:]
                newNeuron.output = self.output[m,k*self.__numOutPerNeuron:(k+1)*self.__numOutPerNeuron,:]
                newNeuron.helpVector = self.helpVector[k*self.__numOutPerNeuron:(k+1)*self.__numOutPerNeuron,:]
                newNeuron.activationFunction = self.activationFunction
                newNeuron.derivativeActivationFunction = self.derivativeActivationFunction
                newNeuron.setState(self.layer,k,m, self.isHidden)
                newNeuron.setParams([self.rate,self.reg])
                lst.append(newNeuron)
            self._neurons.append(lst)

        self.__hasDataSet = True    

    def feedforward_Neuron(self):
        """
        Feedforward for whole layer derived per Neuron
        """
        if not self.__hasDataSet:
            raise ValueError('Warning: Layer Neurons have not been initialized yet')
        for m in range(1):  #run over feature maps
            for k in range(self.__numNeuron): # run over neurons
                self._neurons[m][k].feedforward()

    def feedforward_Layer(self):
        """
        Feedforward for whole layer derived on layer data set
        """
        if Layer.__DEBUG_Layer: print("In feedforward_Layer")
        Neuron.feedforward(self)

    def feedforward(self,version=None):
        """
        Feedforward for whole layer
        """
        if version is None:
            version = Layer.__perNeuron
        if version:
            self.feedforward_Neuron()
        else:
            self.feedforward_Layer()

            
    def backprop(self):
        """
        Derive Layer error vector by backprop pass
        """
        if not self.isForwardPass:
            self.feedforward()    

        #Derive error from parent error
        if (not self.isHidden):   #This is a neuron in the output layer so parent error is the test label set y value
            #self.error[:] = -(self.trainingSet.ydata - self.output[0,:])*self.derivativeActivationFunction(self.helpVector)
            self.error[:] = self._costFunc.dCost(self.output[0,:],self.trainingSet.ydata)*self.derivativeActivationFunction(self.helpVector)
        else:
            self.error[:] = np.dot(self._nextLayer.weights,self._nextLayer.error[0,:])*self.derivativeActivationFunction(self.helpVector)

        # Derive the gradients 
        self._gradCostWeights = np.dot(self.features,self.error[0,:].transpose())/self.trainingSet.numExamples + self.reg*self.weights
        if self.hasBias:
            self._gradCostBias = np.sum(self.error[0,:],axis=1)/self.trainingSet.numExamples

            
        
    def updateWeights(self):
        """
        Update the weight matrix using the gradients from backprop
        """
        self.weights -= self.rate*self._gradCostWeights
        if self.hasBias:
            self.bias[:,0] -= self.rate*self._gradCostBias
        
######
# End class Layer
########################################################################

if __name__ == "__main__":
    import time
    from copy import deepcopy
    nofNeurons = 10
    nofSamples = 2000
    nofFeatures = 100
    nofOutputs = 1  # out of single neuron has to be ONE
    features = np.random.rand(nofFeatures,nofSamples)
    weights = np.random.rand(nofFeatures,nofOutputs*nofNeurons)
    output = np.zeros((1,nofOutputs*nofNeurons,nofSamples))
    helpVector = np.zeros((nofOutputs*nofNeurons,nofSamples))
    error = np.zeros((1,nofOutputs*nofNeurons,nofSamples))
    labels = np.zeros((1,nofOutputs*nofNeurons,nofSamples))
    labels[0,:,:] = np.random.rand(nofOutputs*nofNeurons,nofSamples)
   
    newLayer = Layer([nofSamples,nofNeurons, nofFeatures, nofFeatures, nofOutputs])
    newLayer.setState(1,1)
    newLayer.features = features
    newLayer.weights = weights
    newLayer.output = output
    newLayer.error = error
    newLayer.helpVector = helpVector
    newLayer.activationFunction = np.tan
    newLayer.derivativeActivationFunction = lambda x: 1/np.cos(x)**2
    newLayer.initializeLayer()
    start = time.time()
    newLayer.feedforward(True)
    end = time.time()
    print("Time by perNeuron computation:" + str(end-start))
    out1 = newLayer.output.copy()

    start = time.time()
    newLayer.feedforward(False)
    end = time.time()
    print("Time by perLayer computation:" + str(end-start))
    
    out2 = newLayer.output.copy()
    tst = np.nonzero(out1!=out2)
    #print(tst)
    idx = np.unique(tst[1])
    #print(idx)
    #print(out1[0,idx,:])
    #print(out2[0,idx,:])

    

    newLayer.backprop()
    #newLayer.backprop(labels,False)
    print(newLayer.error)
    #print(newLayer.features)
