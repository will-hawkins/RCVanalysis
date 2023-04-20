import numpy as np
import matplotlib.pyplot as plt

def gen_voters_random(n,d):
	voters = np.random.random((n,d))
	return voters

def gen_voters_tilt(n,d):
    

#voters = gen_voters(10,4)

def calc_distance_tensor(V,C,I=None):
    """
    
    returns
        distances[i,j,k]: distance from voter i to candidate j on issue k.
        
    """
    n,d = V.shape
    c,_ = C.shape
    
    if I is None:
        I = np.ones((n,d)) / d

    distances = np.empty((c,n,d))

    for i in range(c):
        distances[i] = V - C[i]

    return distances# np.multiply(distances, I)

def calc_distance_matrix(V,C,I=None):
    return np.linalg.norm(calc_distance_tensor(V,C,I),axis=2).T