import torch.nn as nn
import torch.optim
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def print_with_time(comments):
    print (comments +"  "+ str(time.asctime( time.localtime(time.time()) )))


def plot_loss_and_acc(loss_and_acc_dict):
	fig = plt.figure()
	tmp = list(loss_and_acc_dict.values())
	maxEpoch = len(tmp[0][0])
	stride = np.ceil(maxEpoch / 10)

	maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
	minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

	for name, lossAndAcc in loss_and_acc_dict.items():
		plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.xticks(range(0, maxEpoch + 1, 2))
	plt.axis([0, maxEpoch, minLoss, maxLoss])
	plt.savefig("./loss_fig.jpg")


	maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
	minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

	fig = plt.figure()

	for name, lossAndAcc in loss_and_acc_dict.items():
		plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.xticks(range(0, maxEpoch + 1, 2))
	plt.axis([0, maxEpoch, minAcc, maxAcc])
	plt.legend()
	plt.savefig("./acc_fig.jpg")
