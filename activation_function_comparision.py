import numpy as np
import time
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Activation_Class:
	def __init__(self):
		self.names = ['sigmoid', 'relu', 'tanh', 'leaky_relu']
		self.colors = [str(name) for name in mcolors.BASE_COLORS.keys()]

	def forward(self, x, name, leaky_slope=0.01):
		if name is not None:
			if name == "sigmoid":
				x = 1 / (1 + np.exp(-x))
			elif name == "relu":
				x = np.maximum(x, 0)
			elif name == "tanh":
				x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
			elif name == "leaky_relu":
				x = np.maximum(x, leaky_slope * x)
			elif name =="softmax":
				x = np.exp(x) / np.sum(np.exp(x),axis=1).reshape(-1,1)
		return x
	
	def backward(self, x, name, leaky_slope=0.01):
		if name == "sigmoid":
			grad = x * (1 - x)
		elif name == "relu":
			grad = (x > 0) * 1
		elif name == "leaky_relu":
			x[x > 0] = 1
			x[x <= 0] = leaky_slope
			grad = x
		elif name =="tanh":
			grad = 1 - (x**2)
		elif name == "softmax":
			grad = x * (1 - x)
		return grad

	def plot(self, x, y_forward, y_backward, color):
		self.axs[0].plot(x, y_forward, color=color)
		self.axs[1].plot(x, y_backward, color=color)
	
	def create_figure(self):
		fig, axs =  plt.subplots(1,2)
		axs[0].set_title("Forward")
		axs[1].set_title("Backward")
		axs[0].set_xlabel("Input")
		axs[0].set_ylabel("Output")
		axs[1].set_xlabel("Input")
		axs[1].set_ylabel("Output")
		return fig, axs

	def run(self):
		input = np.arange(-5,5,0.1)
		for i, name in enumerate(self.names):
			self.fig, self.axs = self.create_figure()
			self.fig.suptitle(name, fontsize=16)
			mean_time = []
			for _ in range(100):
				time_start_forward = time.time()
				output_forward = self.forward(np.array(input), name)
				time_forward_end = time.time()
				output_backward = self.backward(copy.deepcopy(output_forward), name)
				time_end_backward = time.time()
				mean_time.append([time_forward_end - time_start_forward, time_end_backward - time_forward_end])
			print(name + " Forward Time: {}, Backward Time: {}".format(np.mean(mean_time,axis=0)[0], np.mean(mean_time,axis=0)[1]))
			self.plot(input, output_forward, output_backward, self.colors[i])
		plt.show()


if __name__ == "__main__":
	a = Activation_Class()
	a.run()

