##  Copyright (C) 2017 J. Schurer
##  NeuralNet is a Maschine Learning toolkit exploiting neural networks
##
##  This file, test.py, is part of NeuralNet and is for playing 
##  $Id: test.py

import neuralNet.Neuron
import neuralNet.Net
import data.Samples
import numpy as np
import matplotlib.pyplot as plt 

def DisplayData(X, example_width):
    #return [h, display_array] = 


    # Compute rows, cols
    [n, m] = X.shape;
    example_height = n // example_width;

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)));
    display_cols = int(np.ceil(m / display_rows));

    # Between images padding
    pad = 1;

    # Setup blank display
    display_array = - np.ones((int(pad + display_rows * (example_height + pad)), int(pad + display_cols * (example_width + pad))));

    # Copy each example into a patch on the display array
    curr_ex = 0;
    for j in range(1,display_rows+1):
        for i in range(1,display_cols+1):
            if curr_ex >= m:
                break

            #% Copy the patch

            #% Get the max value of the patch
            max_val = np.max(np.abs(X[:,curr_ex]))

            display_array[pad + (j - 1) * (example_height + pad) : (j) * (example_height + pad), \
			              pad + (i - 1) * (example_width  + pad) : (i) * (example_width  + pad)] = \
						   np.reshape(X[:,curr_ex], (example_height, example_width)) / max_val;
            curr_ex = curr_ex + 1;
        if curr_ex > m:
            break

    plt.imshow(display_array.transpose(),cmap=plt.cm.Spectral)

    plt.title("Input Data")
    plt.show()


