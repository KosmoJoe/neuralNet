##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, CostFunctions.py, is part of NeuralNet and contains the class 
##  which implements the possible cost functions
##  $Id: CostFunctions.py

import numpy as np


#No underscore: it's a public variable.
#One underscore: it's a protected variable.
#Two underscores: it's a private variable. 

########################################################################
# Begin class CostFunctions
######
class CostFunctions(object):
    """
    Define a CostFunctions of a neuronal Network
    """

##### Class atrributes
    # type = "Neuron"
    __DEBUG_CostFunctions = False

    def __init__(self,name):
        """
        Initialize a Samples:
        Input: structure = [numFeatures, numExamples, numClasses]
        """
        self.__isInitialized = False                  # Flag to note if all parameter are set
        
        self.__name = name                     # Name of the cost function
        self.__funcNum = -1                     # Number of cost function
        if self.__name == "LeastSquare":
            self.__funcNum = 0
        elif self.__name == "Entropy":
            self.__funcNum = 1
        else:
            raise ValueError("Cost function not supported yet")
        
###################
# Class properties
###################

    #### Constants
    @property
    def name(self):
        return self.__name    


    
    def cost(self):
        """
        Derive the costfunction
        """
        pass


    def dCost(self):
        """
        Derive the derivative of the cost function
        """
        pass
######
# End class CostFunctions
########################################################################


########################################################################
# Begin class LeastSquare
######
class LeastSquare(CostFunctions):
    """
    Define a LeastSquare cost function of a neuronal Network
    """
    def __init__(self):
        super().__init__("LeastSquare")

    def cost(self, prediction, labels):
        """
        Derive the costfunction
        """
        dy = prediction - labels
        return 0.5*np.sum(dy**2)/labels.shape[1]


    def dCost(self, prediction, labels):
        """
        Derive the derivative of the cost function
        """
        return prediction - labels
######
# End class CostFunctions
########################################################################

########################################################################
# Begin class Entropy
######
class Entropy(CostFunctions):
    """
    Define a Entropy cost function of a neuronal Network
    """
    def __init__(self):
        super().__init__("Entropy")

    def cost(self, prediction, labels):
        """
        Derive the costfunction
        """
        return - np.sum(labels*np.log(prediction) + (1-labels)*np.log(1-prediction) )/labels.shape[1]


    def dCost(self, prediction, labels):
        """
        Derive the derivative of the cost function
        """
        return   (prediction - labels)/(prediction*(1-prediction))
######
# End class CostFunctions
########################################################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import scipy.io as sio
    from DisplayData import DisplayData
    newCost = Entropy()
    print(newCost.name)
    print(newCost.dCost(np.random.rand(2,4),np.random.rand(2,4)))