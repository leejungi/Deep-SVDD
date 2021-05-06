import numpy as np
import torch
import torchsummary
import torch.nn as nn
from tqdm import tqdm
import logging
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve


class AE_Trainer:
	def __init__(self, args):
		self.device = args.device
		self.epochs = args.ae_epochs
		self.batch_size = args.batch_size
		self.learning_rate = args.ae_learning_rate
		self.weight_decay = args.ae_weight_decay
		self.criterion = nn.MSELoss().to(self.device)
		self.train_loader = None
		self.test_loader = None
	
	def set_train_loader(self, loader):
		self.train_loader = loader

	def set_test_loader(self, loader):
		self.test_loader = loader


	def train(self, model):
		loader = self.train_loader
		model = model.to(self.device)

		logger = logging.getLogger()
		logger.info("Starting Autoencoder Training")
		total_batch = len(loader)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

		for epoch in tqdm(range(1,self.epochs+1)):
			model.train()
			avg_cost =0
			
			for batch_idx, (X,_) in tqdm(enumerate(loader), total=total_batch):
				X = X.to(self.device)

				optimizer.zero_grad()
				hypothesis = model(X)
				
				scores= torch.sum((hypothesis - X)**2, dim=tuple(range(hypothesis.dim())))
				cost = torch.mean(scores)
				cost.backward()
				optimizer.step()
				avg_cost += cost/total_batch
			scheduler.step()
			logger.info("Epoch: {}/{} \t Loss: {}".format(epoch, self.epochs, avg_cost))

		logger.info("Ending Autoencoder Training")
		torch.save(model.state_dict(), 'save_model/'+str(model.name)+'.pth')

		return model
	
	def test(self, model):
		loader = self.test_loader
		model = model.to(self.device)
		model.eval()
		with torch.no_grad():
			total_batch = len(loader)

			optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

			avg_cost =0
			for batch_idx, (X,Y) in tqdm(enumerate(loader), total=total_batch):
				X = X.to(self.device)
				Y = Y.to(self.device)

				hypothesis = model(X)
				
				cost = self.criterion(hypothesis, Y)
				avg_cost += cost/total_batch
		return Y.to('cpu').numpy(), hypothesis.to('cpu').numpy()

class CNN_Trainer:
	def __init__(self, args):
		self.device = args.device
		self.epochs = args.cnn_epochs
		self.batch_size = args.batch_size
		self.learning_rate = args.cnn_learning_rate
		self.weight_decay = args.cnn_weight_decay
		self.test_interval = args.test_interval
		self.nu = args.nu
		self.criterion = nn.MSELoss().to(self.device)
		self.svdd_mode = args.svdd_mode #"one-class" #"soft"
		self.normal_class = args.normal_class
		self.train_loader = None
		self.test_loader = None
	
	def set_train_loader(self, loader):
		self.train_loader = loader

	def set_test_loader(self, loader):
		self.test_loader = loader


	def train(self, ae_model, model):
		model = self.set_CNN(ae_model,model).to(self.device)

		logger = logging.getLogger()
		logger.info("Starting CNN Training")

		self.c = self.set_c(model, self.train_loader)
		self.R = torch.tensor(0, device=self.device) 
		

		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
		total_batch = len(self.train_loader)

		for epoch in tqdm(range(1,self.epochs+1)):
			model.train()
			avg_cost =0

			
			for batch_idx, (X,Y) in tqdm(enumerate(self.train_loader), total=total_batch):
				X = X.to(self.device)
				Y = Y.to(self.device)

				optimizer.zero_grad()
				hypothesis = model(X)
				
				dist = torch.sum((hypothesis -self.c)**2, dim=1)
		
				if self.svdd_mode == "soft":
					score = dist - self.R**2
					cost = self.R**2 + (1/self.nu)*torch.mean(torch.max(torch.zeros_like(score),score))
				else:
					cost = torch.mean(dist)

				cost.backward()
				optimizer.step()
				avg_cost += cost/total_batch

			logger.info("Epoch: {}/{} \t Loss: {}".format(epoch, self.epochs, avg_cost))
			scheduler.step()

#				if self.svdd_mode == "soft" and epoch >=10:
			if epoch % 5 == 0 and epoch >= 10:
				T_R = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1-self.nu)
				self.R.data = torch.tensor(T_R, device=self.device)


			if epoch % self.test_interval == 0:
				_ = self.test(self.R, self.c, model)


		logger.info("Ending CNN Training")
		torch.save({"R": self.R, "C": self.c, "model":model.state_dict()}, 'save_model/'+str(model.name)+'.tar')

		return self.R, self.c, model
	
	def test(self, R, C, model):	
		model = model.to(self.device)
		logger = logging.getLogger()

		acc_sum =0
		score_list = []
		label_list = []

		model.eval()
		with torch.no_grad():
			total_batch = len(self.test_loader)


			for batch_idx, (X,Y) in tqdm(enumerate(self.test_loader), total=total_batch):
				X = X.to(self.device)
				Y = Y.to(self.device)

				hypothesis = model(X)
					
				dist = torch.sum((hypothesis -C)**2, dim=1)
		
				if self.svdd_mode == "soft":
					score = dist - R**2
					pred = score
				else:
					score = dist
					pred = dist -  R**2
				pred = torch.where(pred<=0,0,1)
				acc = pred == Y

				acc_sum +=  acc.float().sum().item()

				score_list += score.to('cpu')
				label_list += Y.to('cpu')

		accuracy = acc_sum *100.0/(total_batch*self.batch_size)
		score_list = np.array(score_list)
		label_list = np.array(label_list)
		auc = roc_auc_score(label_list, score_list)
		_,_,self.thresholds = roc_curve(label_list, score_list)

		self.thr, acc = self.cal_thr(label_list, score_list, self.thresholds)

		logger.info("Test Accuracy {}% with R {}".format(accuracy, self.R))
		logger.info("Test Accuracy {}% with Thr {}".format(acc, self.thr))
		logger.info("AUC {}%".format(auc*100))
		
		return accuracy

	def cal_thr(self,labels, scores, thresholds):
		pre_acc = 0
		for t in thresholds:
			score = np.where(scores<t,0,1)
			t_acc = score == labels
			t_acc = np.sum(t_acc)/len(score)*100
			if t_acc > pre_acc:
				thr = t
				acc = t_acc
			pre_acc = t_acc
		return thr, acc


	def set_c(self, model, loader):
		c = torch.zeros(model.rep_dim, device=self.device)

		model.eval()
		with torch.no_grad():
			total_batch = len(loader)
			for batch_idx, (X,Y) in enumerate(loader):
				X = X.to(self.device)

				hypothesis = model(X)
				c += torch.sum(hypothesis, dim=0).detach()
		c /= total_batch*self.batch_size

		c[(abs(c) < 0.1) & (c <0)] = -0.1
		c[(abs(c) < 0.1) & (c >0)] = 0.1

		return c
	
			
	def set_CNN(self, ae_model, model):
		ae = ae_model.state_dict()
		cnn = model.state_dict()

		ae = {k:v for k,v in ae.items() if k in cnn}

		cnn.update(ae)
		model.load_state_dict(cnn)
		return model


