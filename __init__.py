##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, __init__.py, is part of NeuralNet and contains 
##  the packages properties
##  $Id: __init__.py


#import NeuralNet

#from QDTK.Wavefunction import Wavefunction as Wfn


########################################################################
## NeuralNet package informations
#########

__author__ = "Johannes Schurer"
__copyright__ = "Copyright 2017, Johannes Schurer"
__license__ = "unset"
__version__ = "0.1"
__version_info__ = tuple(map(int,__version__.split('.')))
__maintainer__ = "Johannes Schurer"
__email__ = "schurer.johannes@gmail.com"



########################################################################
## NeuralNet package functions
#########

def get_path_Base():
    p = NeuralNet.__path__
    l=p[0].split('/')
    return '/'.join(l[:-1])

def get_path_Bin():
    p = get_path_Base()
    return p + '/Bin'

def get_path_Examples():
    p = get_path_Base()
    return p + '/Examples'

