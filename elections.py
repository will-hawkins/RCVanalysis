import numpy as np
from voters import calc_distance_matrix, calc_distance_tensor
class Borda:
    def __init__(self, E):
        """
        args
            votes: 
                array with each row being a ballot, each column
                an alternative. entry i,j is what rank voter i gave 
                to candidate j.
        """

        self.alternatives = E.C
        self.votes = E.to_ballot()
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
        self.num_voters = n
        self.num_alts = d
        self.alt_names = E.alt_names
    def run(self, use_names=False, return_scores=False):
        """
        returns
            returns the order the alternatives finish in.
        """
        scores = np.zeros(self.num_alts, dtype=int)
        for c in range(self.num_alts):
            for r in range(self.num_alts):
                scores[c] += sum(self.votes[:, c] == r) * self.scoring[r]
        results = {self.alt_names[i] : scores[i] for i in range(self.num_alts)}
        return scores, results, (-scores).argsort()
        
class SRB:
    def __init__(self, E):
        self.alternatives = E.C
        self.votes = E.to_ballot()
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
        self.num_voters = n
        self.num_alts = d
        self.alt_names = E.alt_names

    def run(self):
        dictator = np.random.randint(0, self.num_voters)
        return None, None, [self.votes[dictator].argmin()]
class IRV:
    def __init__(self,E):

        self.alternatives = E.C
        self.votes = E.to_ballot()
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
        self.num_voters = n
        self.num_alts = d
        self.alt_names = E.alt_names
        
    def run(self):
        cands = [i for i in range(self.num_alts)]

        won = False
        votes = np.array(self.votes, dtype=np.float64)
        r = 0
        rounds = []
        while not won:
            #breakpoint()
            scores = np.zeros(self.num_alts)
            r+=1
            #print(r)
            R = np.argsort(votes, axis=1)
            #print(R)

            #calculate scores
            for c in cands:
                scores[c] += sum(R[:,0] == c)

            rounds.append(scores)
            #breakpoint()

            if scores.max() > self.num_voters //2:
                results = {self.alt_names[i] : scores[i] for i in range(self.num_alts)}
                #breakpoint()
                return scores, results, (-scores).argsort()

            #breakpoint()
            try:
                last = cands[scores[cands].argmin()]
            except ValueError as E:
                print(E)
                breakpoint()
                pass

            cands.remove(last)
            #votes[votes == last] = np.inf

            #last = R.argmin()
            
            #breakpoint()
            try:
                votes[:,last] = float('inf')
            except:
                breakpoint()
            if r > self.num_alts:
                breakpoint()
                raise Exception('Broken')
    
class Plurality:
    def __init__(self,E):
        self.alternatives = E.C
        self.votes = E.to_ballot()
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
        self.num_voters = n
        self.num_alts = d
        self.alt_names = E.alt_names
    def run(self):
        R = np.argsort(self.votes, axis=1)
        scores = np.zeros(self.num_alts)
        cands = [i for i in range(self.num_alts)]
        for c in cands:
            scores[c] += sum(R[:,0] == c)
        return None, None, (-scores).argsort()
        
class SRC:
    def __init__(self, E):
        self.alternatives = E.C
        self.votes = E.to_ballot()
        n,d = self.votes.shape
        self.scoring = np.arange(d-1,-1,-1)
        self.num_voters = n
        self.num_alts = d
        self.alt_names = E.alt_names
    def run(self):
        return None, None, sorted([i for i in range(self.num_alts)], key=lambda x: np.random.random())

class Election:
    """
    Class to store information about
    candidates and voters for a set of issues.
    """
    def __init__(self, V, C, I=None, alt_names=None):
        self.V = V
        self.C = C
        self.I = I
    
        self.N, self.d = V.shape
        self.A = C.shape[0]
        assert(C.shape[1] == self.d)
        self.alt_names = alt_names
        if alt_names is None:
            self.alt_names = [chr(i+65) for i in range(self.A)]


    def preprocess(self):
        self.dist_matrix = calc_distance_matrix(self.V, self.C,self.I)
        self.dist_tensor = calc_distance_tensor(self.V, self.C,self.I)
        
    def dist_to_alt(self, c):
        #breakpoint()
        return self.dist_matrix[:,c]

    def to_ballot(self):
        ballots = self.dist_matrix.argsort(axis=1)
        #named_ballots = ballots.astype(object).copy()

        return ballots