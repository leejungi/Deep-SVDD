import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class MNIST_CNN(nn.Module):
	def __init__(self, rep_dim=32):
		super(MNIST_CNN, self).__init__()
		self.name = "MNIST_CNN"
		self.rep_dim = rep_dim
		self.pool = nn.MaxPool2d(2,2)

		#Encoder
		self.encoder = nn.Sequential(
					nn.Conv2d(1,8,5,bias=False, padding=2),
					nn.BatchNorm2d(8,eps=1e-04, affine=False),
					nn.LeakyReLU(),
					nn.MaxPool2d(2,2),

					nn.Conv2d(8,4,5,bias=False, padding=2),
					nn.BatchNorm2d(4,eps=1e-04, affine=False),
					nn.LeakyReLU(),
					nn.MaxPool2d(2,2)
				)

		#Representation layer
		self.fc1= nn.Linear(4*7*7, self.rep_dim, bias=False)

	def forward(self,x):
		#Encoder
		x = self.encoder(x)

		#Representation
		x = x.view(x.size(0), -1)
		x = self.fc1(x)

		return x


class MNIST_AE(nn.Module):
	def __init__(self, rep_dim=32):
		super(MNIST_AE, self).__init__()
		self.name = "MNIST_AE"
		self.rep_dim = rep_dim
		self.pool = nn.MaxPool2d(2,2)

		#Encoder
		self.encoder = nn.Sequential(
					nn.Conv2d(1,8,5,bias=False, padding=2),
					nn.BatchNorm2d(8,eps=1e-04, affine=False),
					nn.LeakyReLU(),
					nn.MaxPool2d(2,2),

					nn.Conv2d(8,4,5,bias=False, padding=2),
					nn.BatchNorm2d(4,eps=1e-04, affine=False),
					nn.LeakyReLU(),
					nn.MaxPool2d(2,2)
				)

		#Representation layer
		self.fc1= nn.Linear(4*7*7, self.rep_dim, bias=False)

		#Decoder
		self.deconv1 = nn.ConvTranspose2d(2,4,5, bias=False, padding=2)
		self.bn1 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
		self.deconv2 = nn.ConvTranspose2d(4,8,5, bias=False, padding=3)
		self.bn2 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
		self.deconv3 = nn.ConvTranspose2d(8,1,5, bias=False, padding=2)

	def forward(self,x):
		#Encoder
		x = self.encoder(x)

		#Representation
		x = x.view(x.size(0), -1)
		x = self.fc1(x)

		#Decoder
		x = x.view(x.size(0), int(self.rep_dim/16), 4, 4)
		x = F.interpolate(F.leaky_relu(x), scale_factor=2)
		x = self.deconv1(x)
		x = F.interpolate(F.leaky_relu(self.bn1(x)), scale_factor=2)
		x = self.deconv2(x)
		x = F.interpolate(F.leaky_relu(self.bn2(x)), scale_factor=2)
		x = self.deconv3(x)
		x = torch.sigmoid(x)

		return x

if __name__=="__main__":
	AE = MNIST_AE()
	print(torchsummary.summary(AE, (1,28,28), device='cpu'))
