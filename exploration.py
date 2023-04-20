import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


from elections import *
from voters import *

matplotlib.use('WebAgg')

"""P = gen_voters(10,2)
A = gen_voters(3,2)

E = Election(P,A)

E.preprocess()

fig,ax = plt.subplots()
ax.scatter(P[:,0], P[:,1], c='r')
ax.scatter(A[:,0], A[:,1],c='b')

for i, txt in enumerate(['a','b','c']):
	ax.annotate(txt, (A[i,0], A[i,1]))

print(E.to_ballot())
print(Borda(E).run())
plt.show()"""

def sim(num_sim, N, A, d, method):
	dist_to_winner = np.zeros([num_sim,2])
	opt_winner = np.zeros(num_sim)
	np.random.seed(40)
	for i in range(num_sim):
		P = gen_voters(N,d)
		C = gen_voters(A,d)

		E = Election(P, C)
		E.preprocess()

		s, r, o = method(E).run()


		dist_to_winner[i] = E.dist_to_alt(o[0]).mean(), np.median(E.dist_to_alt(o[0]))

		opt_winner[i] = int(E.dist_matrix.sum(axis=0).argmin() == o[0])

	#breakpoint()
	return dist_to_winner, opt_winner

if __name__ == '__main__':
	data = {}
	V = [10,50,100,500,1000,5000,10000]
	C =[2,3,4,5,10]
	sim(1,10,3,2,SRC)
	#breakpoint()
	methods = [IRV,SRB,Borda, SRC, Plurality]
	n = len(methods)
	d = 3
	
	for ax,method in enumerate(methods):
		mean = np.zeros([len(V),len(C)])
		opt = np.zeros([len(V),len(C)])
		med = np.zeros([len(V),len(C)])
		for i,v in enumerate(V):
			for j,c in enumerate(C):
				D,O = sim(1000, v, c, 2, method)
				mean[i,j] = D[:,0].mean()
				med[i,j]  = D[:,1].mean()
				opt[i,j] = O.mean()

		print(opt)
		data[method.__name__] = mean,opt,med
		
	fig, axs = plt.subplots(nrows=3, ncols=n+1, figsize=(n*3,d*3), width_ratios=[3]*n+[.1])


	vmin = min([df[0].min() for df in data.values()])
	vmax = max([df[0].max() for df in data.values()])
	i = 0
	for method in methods:
		dat = data[method.__name__][0]

		dab = pd.DataFrame(dat, columns=C, index=V)	
		sns.heatmap(dab, annot=False, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[0][i], cbar=False)
		axs[0][i].set_title(method.__name__)	
		i+=1
	fig.colorbar(axs[0][0].collections[0], cax=axs[0][-1])


	vmin = min([df[1].min() for df in data.values()])
	vmax = max([df[1].max() for df in data.values()])
	i = 0
	for method in methods:
		dat = data[method.__name__][1]

		dab = pd.DataFrame(dat, columns=C, index=V)	
		sns.heatmap(dab, annot=False, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[1][i], cbar=False)
		
		i+=1
	fig.colorbar(axs[1][0].collections[0], cax=axs[1][-1])


	vmin = min([df[2].min() for df in data.values()])
	vmax = max([df[2].max() for df in data.values()])
	i = 0
	for method in methods:
		dat = data[method.__name__][2]

		dab = pd.DataFrame(dat, columns=C, index=V)	
		sns.heatmap(dab, annot=False, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[2][i], cbar=False)
		
		i+=1
	fig.colorbar(axs[2][0].collections[0], cax=axs[2][-1])
	

	fig.tight_layout()
	fig.savefig(f"figure.png")