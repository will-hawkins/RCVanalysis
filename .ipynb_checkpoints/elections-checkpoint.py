import numpy as np
from voters import calc_distance_matrix, calc_distance_tensor
class Borda:
    def __init__(self, votes, alternatives):
        self.votes = votes
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
    def run(self):
        n,d = self.votes.shape
        scores = np.zeros(d, dtype=int)
        for i in range(d):
            R = np.unique(self.votes[:,i], return_counts=True)
            scores += R[1] * self.scoring[i]
        return scores
        
        
class IRV:
    def __init__(self,votes, alternatives):
        self.votes = np.array(votes, dtype=object)
        
    def run(self):
        won = False
        votes = self.votes
        n,d = votes.shape
        r = 0
        while not won:
            r+=1
            #print(r)
            R = np.unique(votes[:,0], return_counts=True)
            #print(R)
            if R[1].max() > n //2:
                return R[1].argmax()
            last = R[1].argmin()
            elim = votes[:,0] == last
            out = votes == last
            votes = votes[~out].reshape(n,d-r)
            
        return R
    
class Election:
    def __init__(self, V, C, I=None):
        self.V = V
        self.C = C
        self.I = I
    
    def preprocess(self):
        self.dist_matrix = calc_distance_matrix(self.V, self.C,self.I)
        self.dist_tensor = calc_distance_tensor(self.V, self.C,self.I)
        
    def dist_to_alt(self, c):
        return self.dist_matrix[:,c].sum()