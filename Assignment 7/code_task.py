import numpy as np
import matplotlib.pyplot as plt
import math

def weighted_majority(d, T, n=100):
	"""
	d -- number of experts
	T -- number of rounds
	"""
	nu = math.sqrt(2*math.log(T)/d)
	w = np.ones(d)
	for t in range(1,T+1):
		v = []
		for i in range (1, d+1):
			cost_i = np.random.binomial(n, 1-(i/(2*d)), d)/n
			v.append(cost_i)
		np_v = np.array(v)
		cost = sum(v*w)
		w = [x * cost for x in w]
	return w


def plot_loss(loss, round, d, T):
	plt.plot(loss, round, 'b.')
	plt.title("cumulative loss, d="+d+",T="+T)
	plt.xlabel("round")
	plt.ylabel("loss")
	# plt.grid()
	plt.legend()
	plt.savefig("cumulative_loss.png")
	plt.show()
	plt.clf()


def plot_regret(regret, round, d, T):
	plt.plot(regret, round, 'b.')
	plt.title("regret of the weighted majority algorithm, d="+d+",T="+T)
	plt.xlabel("round")
	plt.ylabel("loss")
	# plt.grid()
	plt.legend()
	plt.savefig("regret.png")
	plt.show()
	plt.clf()


print(weighted_majority(d=10, T=100))
#plot_loss(loss, round d=10, T=100)

