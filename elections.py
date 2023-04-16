import numpy as np
from voters import calc_distance_matrix, calc_distance_tensor
class Borda:
    def __init__(self, votes, alternatives):
        """
        args
            votes: 
                array with each row being a ballot, each column
                an alternative. entry i,j is what rank voter i gave 
                to candidate j.
        """
        self.votes = votes
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
        self.num_voters = n
        self.num_alts = d
        self.alt_names = alternatives
    def run(self, use_names=True):
        """
        returns
            returns the order the alternatives finish in.
        """
        scores = np.zeros(d, dtype=int)
        for i in range(d):
            R = np.unique(self.votes[:,i], return_counts=True)
            scores += R[1] * self.scoring[i]
        if use_names:
            re
            return result
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

    def to_ballot(self):
        pass