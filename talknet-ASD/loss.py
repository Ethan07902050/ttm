import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
		
	def forward(self, x, labels=None):	
		if labels == None:
			predScore = x[:,1]
			predScore = F.softmax(x, dim = -1)
			# predScore = predScore.t()
			# predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			nloss = self.criterion(x, labels)
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum

class lossA(nn.Module):
	def __init__(self):
		super(lossA, self).__init__()
		self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

	def forward(self, x, labels):	
		nloss = self.criterion(x, labels)
		return nloss

class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()
		self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

	def forward(self, x, labels):	
		nloss = self.criterion(x, labels)
		return nloss

