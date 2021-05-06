import spectral
from scipy import io
import os
import numpy as np
import h5py
from collections import Counter
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import torch

#	Preprocessing class
class Preprocessing:
	def __init__(self):
		self.scaler = MinMaxScaler()
		self.under_method = ['OSS',
				  'OneSidedSelection']
		self.over_method = ['ADASYN'
				  ]

	def under_sampling(self, data, label, n_neighbors=5, method=None):
		#Input
		#	data: 2D array data (im_height*im_width, num of band)
		#	label: 1D array label(0,1,2...) per each data
		#	n_neighbors: num of neighbors used in OSS
		#	method: select under sampling method (OSS)
		#Output
		#	return under sampled data, label
		if method in self.under_method:
			print("Before sampling label proportion: ",Counter(label))
			if method == 'OSS' or method == 'OneSidedSelection':	  
				undersample = OneSidedSelection(n_neighbors=n_neighbors, n_seeds_S=200)
				data, label = undersample.fit_resample(data, label)
				
			print("After sampling label proportion: ",Counter(label))
		
		return data, label


	def over_sampling(self, data, label, n_neighbors=5, sampling_ratio=0.1, method=None):
		#Input
		#	data: 2D array data (im_height*im_width, num of band)
		#	label: 1D array label(0,1,2...) per each data
		#	n_neighbors: num of neighbors used in OSS
		#	sampling_ratio: num of over sampling is determined by max labeled data * sampling_ratio
		#	method: select under sampling method (ADASYN)
		#Output
		#	return over sampled data, label

		if method in self.over_method:
			print("Before sampling label proportion: ",Counter(label))
			if method == 'ADASYN':    
				strategy = Counter(label)
				max_value = max(strategy.values())
				for k,v in strategy.items():
					if v < int(max_value*0.1):
						strategy[k] = int(max_value*sampling_ratio)
				data, label = ADASYN(sampling_strategy=strategy).fit_resample(data,label)
				
			print("After sampling label proportion: ",Counter(label))
		
		return data, label

	def normalize(self, data, fit=True):
#		Input
#			data: 4D array data (num of data, im_height, im_width, num of band)
#				MinMax - use sklearn MinMaxScaler
#			fit: if True tuning Minmax scaler
#		Output
#			return Normalized data

		data = np.array(data)
		original_shape = data.shape 
		data = data.reshape(np.prod(data.shape[:1]), np.prod(data.shape[1:]))
		
		if fit == True:
			self.scaler.fit(data)
			data = self.scaler.transform(data)
			joblib.dump(self.scaler, "save_model/scaler.gz")
		else:
			#Load Scaler: this is for only inference executing(without training)
			self.scaler=joblib.load("save_model/scaler.gz")
			data = self.scaler.transform(data)
		data = data.reshape(original_shape)

		return data

	def PCA(self, data, n_comp, fit=True):
#		Input 
#			data: 2D array data (im_height*im_width, num of band)
#			n_comp: number of components for PCA
#			fit: if True tuing PCA
#		Output
#			return data with dimension reduction

	
		if n_comp !=0:
			if fit == True:

				original_shape = data.shape 
				data = data.reshape(np.prod(data.shape[:3]), np.prod(data.shape[3:]))

				self.pca = PCA(n_components=n_comp)
				self.pca.fit(data)
				data = self.pca.transform(data)
				joblib.dump(self.pca, "save_model/pca.gz")

				data = data.reshape((*original_shape[:3],n_comp))
			else:
				original_shape = data.shape 
				data = data.reshape(np.prod(data.shape[:3]), np.prod(data.shape[3:]))

				self.pca=joblib.load("save_model/pca.gz")
				data = self.pca.transform(data)

				data = data.reshape((*original_shape[:3],n_comp))

#		Ploting pca variance ratio, this is used for selecting number of components
#		plt.plot(self.pca.explained_variance_ratio_)
#		plt.show()
		return data

	def MA(self, data, width):
#		Moving Average
#		Input 
#			data: 2D array data (im_height*im_width, num of band)
#			width: width of moving average
#		Output
#			return data with MA
		if width !=0:
			ma_data = np.zeros((data.shape[0], data.shape[1]-width+1))
			for i, d in enumerate(data):
				#Apply MA
				ma_data[i] = np.convolve(d, np.ones((width,))/float(width), mode='valid')

#		Ploting the Moving average result
#		plt.plot(data[0])
#		plt.plot(ma_data[0])
#		plt.show()
			return ma_data	
		else:
			return data

	#Binarize label
	def binarize(self, label):
	# label<2: 0
	# label>=2: 1
		return np.where(label<2, 0, 1)

	#Remove label
	def ignore_label(self, data, label, ignored_label):
		if ignored_label != []:
			new_data = []
			new_label = []
			for l in np.unique(label):
				if l not in ignored_label:
					indices = np.where(label==l)
					if new_data == []:
						new_data = data[indices]
						new_label = label[indices]
					else:
						new_data = np.vstack([new_data, data[indices]])
						new_label = np.hstack([new_label, label[indices]])
			return new_data, new_label
		else:
			return data, label

def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

