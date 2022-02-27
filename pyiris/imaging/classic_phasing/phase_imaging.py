#for irregularly spaced antenna arrays

import numpy as np

#inputs: Frequency, baselines, resolution, FOV, Visibilities, 

#sum contributions in cartesian space based on visibilities from each baseline

#assume far field (can set distance to 10x far-field equation based on longest baseline)

#synthesize the FOV based on user inputs for angles to calculate

#This should be straightforward
