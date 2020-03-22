import numpy as np

def normal_min_max(x):
    '''
    Returns a list of the same size and order after applying the normalization
    y(j) = ( x(j) - min(x) )/ (max(x) - min(x))
    Then y(j) varies between 0 and 1
    '''
    y = np.zeros(shape=x.size)
    x_min = np.min(x)
    x_max = np.max(x)
    x_span = x_max - x_min
    
    for j in range(x.size):
        y[j] = (x[j] - x_min)/x_span
    
    return y
