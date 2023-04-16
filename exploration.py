import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from elections import *
from voters import *

matplotlib.use('WebAgg')

P = gen_voters(10,2)
A = gen_voters(2,2)

E = Election(P,A)
E.preprocess()
plt.scatter(P[:,0], P[:,1], c='r')
plt.scatter(A[:,0], A[:,1],c='b')

print(self.aE.dist_matrix.argsort(axis=1))
plt.show()

