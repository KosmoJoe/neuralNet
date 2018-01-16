##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, Neuron.py, is part of NeuralNet and contains the class 
##  which implements a single neuron including all its properties
##  $Id: Neuron.py

import numpy as np

#No underscore: it's a public variable.
#One underscore: it's a protected variable.
#Two underscores: it's a private variable. 


########################################################################
# Begin class Neuron
######
class Neuron(object):
    """
    Define a Neuron of a neuronal Network
    """

##### Class atrributes
    # type = "Neuron"
    __DEBUG_Neuron = False

    def __init__(self,dimensions):
        """
        Initialize a neuron:
        Input dimensions = [numBatch, numIn, numWeights, numOut]
        """
        self.__isInitialized = False                  # Flag to note if all parameter are set
        
        self.__numIn = dimensions[1]                  # Number of inputs/features
        #self.__numInMaps = dimensions[1]              # Number of feature maps
        self.__numOut = dimensions[3]                 # Number of output values (num of out neurons)
        #self.__numOutMaps = dimensions[3]             # Number of output maps
        self.__numWeight = dimensions[2]                 # Number of output values
        self.__numBatch = dimensions[0]                 # Number of samples/ Batch size
        self.__numOutFeatures = 0                      # Number of output features / dimension of the output
        
        if self.__numIn == self.__numWeight:
            self.__convolute = False
            self.__numOutFeatures = 1                  # Single value per neuron to next layer
        else:
            self.__convolute = True
            self.__numOutFeatures = self.__numIn - self.__numWeight + 1

        self.__neuronNum = 0                          # Number of neuron within Layer
        self.__layer = 0                              # Layer number of neuron
        self.__map = 0                                # Map number of neuron
        self.__isHidden = (self.__layer == 0)         # Flag for hidden layers
        self.__shareWeights = False                   # Share the weights within the layer
        self.__isForwardPass = False                  # Flag to test if a forward pass was done


        self._activationFunction = None  # Pointer to activation function f(wt*x + b)
        self._derivativeActivationFunction = None  # Pointer to derivative of activation function f(wt*x + b)

        ## Matrices are stored in row major order: A[i,j]  i: row, j :column
        ## and are MxN with M rows and N columns
        ## Here: row is map index 
        ##       column is element index
        self._features = None            # Pointer to features x[featureIndex,sampleIndex] element (self.__numIn)x(self.__numBatch)
        self._weights = None             # Pointer to weigths  w[featureIndex,outputIndex] element (self.__numWeight)x(self.__numOut)
        self._output = None              # Pointer to output   a[outputIndex,sampleIndex]  element (self.__numOut)x(self.__numOutFeatures)x(self.__numBatch)
                                         #                                                      or (self.__numOut)x(self.__numBatch) for self.__convolute = False

        self.__helpVector = None         # Help Vector z =  (wt*x + b)   or   (wt*x)

        self.__hasBias = False           # Flag to switch between notion of an additional bias or a feature x_0 = 1
        self._bias = None                # Bias term

        self._rate = 0                   # Learning rate
        self._reg = 0                    # Regularization Parameter
        #tape = opts.get("tape",None)
        #if tape: self.init_from_tape(tape)


###################
# Class properties
###################

    #### Constants
    @property
    def numIn(self):
        return self.__numIn
    @property
    def numOut(self):
        return self.__numOut
    @property
    def numWeight(self):
        return self.__numWeight
    @property
    def numBatch(self):
        return self.__numBatch
    @property
    def layer(self):
        return self.__layer
    @property
    def isHidden(self):
        return self.__isHidden
    @property
    def isInitialized(self):
        return self.__isInitialized
    @property
    def isForwardPass(self):
        return self.__isForwardPass

    @property
    def hasBias(self):
        return self.__hasBias
    
    @hasBias.setter
    def hasBias(self, val):
        if isinstance(val, (bool)):
            self.__hasBias = val
        else:
            self._bias = False

    #### Bias
    @property
    def bias(self):
        return self._bias
    
    @bias.setter
    def bias(self, val):
        if isinstance(val, (int, long, float, complex)):
            self._bias = val
            self.__hasBias = True
        else:
            self._bias = None

    #### learning rate
    @property
    def rate(self):
        return self._rate
    
    @rate.setter
    def rate(self, val):
        if isinstance(val, (int, long, float)):
            self._rate = val
        else:
            self._rate = 0

    #### regularization parameter
    @property
    def reg(self):
        return self._reg
    
    @reg.setter
    def reg(self, val):
        if isinstance(val, (int, long, float)):
            self._reg = val
        else:
            self._reg = 0

    #### FEATURES
    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self, val):
        if val.shape == (self.__numIn,self.__numBatch):
            self._features = val
            if Neuron.__DEBUG_Neuron: print("Feature Vector has been set")
        else:
            self._features = None

    #### WEIGHTS
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        if val.shape == (self.__numWeight, self.__numOut):
            self._weights = val
            if Neuron.__DEBUG_Neuron: print("Weights have been set")
        else:
            self._weights = None

    #### OUTPUT
    @property
    def output(self):
        return self._output
        
    @output.setter       ## Output setter to use external memory
    def output(self, val):
        if self.__convolute:
           dims = (self.__numOut, self.__numOutFeatures, self.__numBatch)
        else:
           dims = (self.__numOut, self.__numBatch)
        if val.shape == dims:
            self._output = val
            if Neuron.__DEBUG_Neuron: print("Output is set")
        else:
            self._output = None

    #### Helper Z 
    @property
    def helpVector(self):
        return self.__helpVector
        
    @helpVector.setter       ## helpVector setter to use external memory
    def helpVector(self, val):
        if self.__convolute:
           dims = (self.__numOut, self.__numOutFeatures, self.__numBatch)
        else:
           dims = (self.__numOut, self.__numBatch)
        if val.shape == dims:
            self.__helpVector = val
            if Neuron.__DEBUG_Neuron: print("helpVector is set")
        else:
            self.__helpVector = None

    #### ACTIVATION FUNCTION
    @property
    def activationFunction(self):
        return self._activationFunction

    @activationFunction.setter
    def activationFunction(self, val):
        if callable(val):
            self._activationFunction = val
            if Neuron.__DEBUG_Neuron: print("Activation Function has been set")
        else:
            self._activationFunction = None
            
    @property
    def derivativeActivationFunction(self):
        return self._derivativeActivationFunction

    @derivativeActivationFunction.setter
    def derivativeActivationFunction(self, val):
        if callable(val):
            self._derivativeActivationFunction = val
            if Neuron.__DEBUG_Neuron: print("Derivative of Activation Function has been set")
        else:
            self._derivativeActivationFunction = None


###################
# Class functions
###################

## Internal Functions
    def _apply_weights(self):
        """
        Apply the weights to the input vector
        """
        if Neuron.__DEBUG_Neuron: print("In apply_weights: Neuron")
        if not self.__convolute:
            self.__helpVector = np.dot(self.weights.transpose(),self.features)
        else:
            self.__helpVector = self._2D_convolute(self.features,self.weights)
        if self.__hasBias:
            self.__helpVector += self.bias
            
    def _2D_convolute(self,features,weights):
        """
        Cope with convolutions
        """
        return np.zeros(self.__numOut)

    def feedforward(self):
        """
        Derive Neuron output vector by forward pass
        """
        if Neuron.__DEBUG_Neuron: print("In feedforward Neuron")
        if not self.__isInitialized:
            self._check_initialize()
           
        self._apply_weights()
        self.output[:] = self.activationFunction(self.__helpVector)
        self.__isForwardPass = True
        return self.output



    def _check_initialize(self):
        """
        Check if all needed input was supplied
        """
        if (self.features is None):
            raise ValueError('Warning: No feature vector given')
        if (self.weights is None):
            raise ValueError('Warning: No weights vector given')
        if (self.activationFunction is None):
            raise ValueError('Warning: No activation function given')
        if (self.derivativeActivationFunction is None):
            raise ValueError('Warning: No derivative activation function given')
        
        self.__isInitialized = True

    def __str__(self):
        neuronString = "" + str(self.__neuronNum) + "th Neuron in Layer " + str(self.__layer) + " in map " + str(self.__map) + "\n"
        neuronString += "Feature Vector:   " + str(self.features.shape) + "\n"
        if self.__hasBias: neuronString += "Bias Term:        " + str(self.bias) + "\n"
        neuronString += "Weights:          " + str(self.weights.shape) + "\n"
        neuronString += "Output:           " + str(self.output.shape) + "\n"
        neuronString += "Activation Func:  " + str(self.activationFunction) + ""
        return neuronString

    def setState(self, layer,neuronNum,mapNum, isHidden):
        self.__layer = layer
        self.__neuronNum = neuronNum
        self.__map = mapNum
        self.__isHidden = isHidden


    def setParams(self,params):
        """
        Function to set learning parameters
        """
        self._rate = params[0]
        self._reg  = params[1]

######
# End class Neuron
########################################################################

if __name__ == "__main__":
   from copy import deepcopy
   nofNeurons = 10
   features = np.random.rand(10*nofNeurons,100)
   weights = np.random.rand(10*nofNeurons,2)
   output = np.zeros((1*nofNeurons,100))
   bias = 2.1

   for j in range(nofNeurons):
      myNeuron = Neuron([100,10,10,1])
      myNeuron.weights = weights[j*10:(j+1)*10,:]
      myNeuron.features = features[j*10:(j+1)*10,:]
      myNeuron.output = output[j*2:(j+1)*2,:]
      myNeuron.activationFunction = np.tan
      print(myNeuron)
      outputNeuron = myNeuron.feedforward()
      #print(j, output.shape)
      print(output)

      #myNeuron2 = deepcopy(myNeuron)
   
