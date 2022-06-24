import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
from skimage import filters
import torch.nn.functional as F
import cv2


def getEdge(batch):
	edgeslist=[]
	for kk in range(batch.size(0)):
		x=batch[kk]
		# print(x.size())
		x=x.cpu().data.numpy()
		# if len(x.shape)>2:
		# 	# x=np.transpose(x,(1,2,0))
		# 	# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
		# 	x = x[0]
		edges = filters.sobel(x)
		edgeslist.append(edges)
	edgeslist=np.array(edgeslist)
	edgeslist=torch.Tensor(edgeslist).cuda()
	# edgeslist=F.Variable(edgeslist)
	return  edgeslist


def dice_loss(predict, target):

	smooth = 1e-5

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)
	intersection = torch.sum(torch.mul(y_pred_f, y_true_f), dim=1)
	union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1) + smooth
	dice_score = (2.0 * intersection / union)

	dice_loss = 1 - dice_score

	return dice_loss

def rank_loss(predict, target):
	top_k = 30

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)

	N_topvalue, N_indice = (y_pred_f * (1 - y_true_f)).topk(top_k, dim=-1, largest=True, sorted=True)

	P_values, P_indice = ((1.0 - y_pred_f) * y_true_f).topk(top_k, dim=-1, largest=True, sorted=True)
	P_downvalue = 1 - P_values

	beta = 1
	rank_loss = 0
	for i in range(top_k):
		for j in range(top_k):
			th_value = N_topvalue[:,i] - beta * P_downvalue[:,j] + 0.3
			rank_loss = rank_loss + (th_value * (th_value>0).float()).mean()

	return rank_loss/(top_k * top_k)

class Dice(nn.Module):
	def __init__(self):
		super(Dice, self).__init__()

	def forward(self, predicts, target):

		preds = torch.softmax(predicts, dim=1)
		dice_loss0 = dice_loss(preds[:, 0, :, :], 1 - target)
		dice_loss1 = dice_loss(preds[:, 1, :, :], target)
		loss_D = (dice_loss0.mean() + dice_loss1.mean())/2.0


		return loss_D


class SLS_loss(nn.Module):
	def __init__(self):
		super(SLS_loss, self).__init__()

	def EPE(self, pred_edge, gt_edge, sparse=False, mean=True):
		EPE_map = torch.norm(gt_edge - pred_edge, 2, 1)
		if sparse:
			EPE_map = EPE_map[gt_edge != 0]
		if mean:
			return EPE_map.mean()
		else:
			return EPE_map.sum()

	def nll_loss(self,preds,target):
		NLL= nn.NLLLoss()
		loss = NLL(preds,target)
		return loss

	def forward(self, predicts, target):

		softmax = nn.Softmax(dim=1)
		LogSoftmax = nn.LogSoftmax(dim=1)

		preds = softmax(predicts)
		preds = LogSoftmax(preds)
		target = target.long()
		loss_1 = self.nll_loss(preds,target)

		preds = preds[:,0,:,:]
		pred_edge = getEdge(preds)
		target_edge = getEdge(target)
		loss_2 = self.EPE(pred_edge,target_edge)

		return loss_1, loss_2
















