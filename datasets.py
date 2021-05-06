import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from preprocess import global_contrast_normalization

class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, label, normal=0,  train=True):
		super(Dataset, self).__init__()

		self.data = data
		self.label = label
		self.normal = normal
		# Pre-computed min and max values (after applying GCN) from train data per class
		min_max = [(-0.8826567065619495, 9.001545489292527),
				   (-0.6661464580883915, 20.108062262467364),
				   (-0.7820454743183202, 11.665100841080346),
				   (-0.7645772083211267, 12.895051191467457),
				   (-0.7253923114302238, 12.683235701611533),
				   (-0.7698501867861425, 13.103278415430502),
				   (-0.778418217980696, 10.457837397569108),
				   (-0.7129780970522351, 12.057777597673047),
				   (-0.8280402650205075, 10.581538445782988),
				   (-0.7369959242164307, 10.697039838804978)]

		# MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
		self.transform = transforms.Compose([transforms.ToTensor(),
										transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
										transforms.Normalize([min_max[normal][0]],
															 [min_max[normal][1] - min_max[normal][0]])])




		self.label = np.where(self.label == normal, 0, 1)
		if train==True:
			self.indices = np.where(self.label==normal)
			self.data = self.data[self.indices]
			self.label = self.label[self.indices]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		data = self.data[index]
		data = torch.from_numpy(data).float()
		data = data.unsqueeze(0) 

		# standardization
		mean = torch.mean(data) 
		data -= mean

		x_scale = torch.mean(torch.abs(data))


		data /= x_scale


#		data= Image.fromarray(data.numpy(), mode="L")
#		data= self.transform(data)


		label = self.label[index]
		return data, label 


		

