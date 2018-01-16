##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, Samples.py, is part of NeuralNet and contains the class 
##  which implements a set of sample data
##  $Id: Samples.py

import numpy as np


#No underscore: it's a public variable.
#One underscore: it's a protected variable.
#Two underscores: it's a private variable. 

########################################################################
# Begin class Samples
######
class Samples(object):
    """
    Define a Sample of a neuronal Network
    """

##### Class atrributes
    # type = "Neuron"
    __DEBUG_Samples = False

    def __init__(self,structure):
        """
        Initialize a Samples:
        Input: structure = [numFeatures, numExamples, numClasses]
        """
        self.__isInitialized = False                  # Flag to note if all parameter are set
        
        self.__numFeatures = structure[0]             # Number of features of a single example
        self.__numExamples = structure[1]             # Number of examples in a sample
        self.__numClasses = structure[2]              # Number of classes for the sample label

        self._xdata = None                            # Pointer to input data array[numFeatures,numExamples]
        self._ydata = None                            # Pointer to index of labels of xdata as array[numClasses, numExamples] 
        self._y = None                                # Pointer to labels of data as array[numExamples]

        self._labels = None                             # List of the labels

        
###################
# Class properties
###################

    #### Constants
    @property
    def numFeatures(self):
        return self.__numFeatures    
    @property
    def numExamples(self):
        return self.__numExamples
    @property
    def numClasses(self):
        return self.__numClasses


    @property
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, val):
        if val.shape == (self.__numClasses,) :
            self._labels = val
            if Samples.__DEBUG_Samples: print("Labels have been set")
        else:
            self._labels = None

    @property
    def xdata(self):
        return self._xdata
    @xdata.setter
    def xdata(self, val):
        if val.shape == (self.__numFeatures,self.__numExamples):
            self._xdata = val
            if Samples.__DEBUG_Samples: print("XData has been set")
        else:
            self._xdata = None

    @property
    def ydata(self):
        return self._ydata
    @ydata.setter
    def ydata(self, val):
        if val.shape == (self.__numClasses,self.__numExamples):
            self._ydata = val
            if Samples.__DEBUG_Samples: print("YData has been set")
        else:
            self._ydata = None

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, val):
        if val.shape == (self.__numExamples,1):
            self._y = val
            help_y = np.ones((1,self.__numExamples))*np.arange(1,self.__numClasses+1)[:, np.newaxis]  # Promote to vectors of ones 
            self.ydata = (help_y==self.y.transpose()).astype(int)
            if Samples.__DEBUG_Samples: print("YData has been set")
        else:
            self._y = None




    
    def predict(self):
        """
        Derive the prediction
        """
        pass
######
# End class Samples
########################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import scipy.io as sio
    from DisplayData import DisplayData
    matfile = sio.loadmat('data1.mat')
    numExamples = matfile['y'].shape[0]
    numFeatures = matfile['X'].shape[1]
    numClasses = 10
    #print(matfile['X'].shape)
    #print(matfile['y'])
    newSample = Samples([numFeatures,numExamples, numClasses])
    newSample.xdata = matfile['X'].transpose()
    newSample.ydata = matfile['y']
    newSample.label = range(10)
    print(newSample.xdata)
    
    selection = newSample.xdata.transpose().copy()
    np.random.shuffle(selection)
    selection = selection.transpose()
    
    #DisplayData(newSample.xdata[:,0:5000:100],20)
    DisplayData(selection[:,0:5000:100],20)
    