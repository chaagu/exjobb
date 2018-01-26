# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:44:21 2018

@author: chaagu
"""

import numpy as np
import random 

def random_array ():
    for x in range(1):
        x = random.randint(10,50)   # Generating a random value higher than 10 
                                    # and below 50
        
    array = np.arange(x)            # Create an array with length x
    np.random.shuffle(array)        # Shuffle the array - Random Array
    
    print(array)
    return

random_array()      # Test

