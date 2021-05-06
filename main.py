import numpy as np
import torch
import logging
import argparse
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor 

from datasets import Dataset
from model.MNIST import MNIST_AE, MNIST_CNN
from model.Trainer import AE_Trainer, CNN_Trainer
from preprocess import Preprocessing

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Deep SVDD code reproduced by Jungi Lee", formatter_class=RawTextHelpFormatter)

	#Model params
	parser.add_argument('--device', type=str, default='cpu', help="Device name for torch executing environment e.g) cpu, cuda")
	parser.add_argument('--batch_size', type=int, default=200, help="Batch size")
	parser.add_argument('--ae_epochs', type=int, default=150, help="AutoEncoder Num of Epochs")
	parser.add_argument('--ae_learning_rate', type=float, default=1e-4, help="AutoEncoder Learning Rate")
	parser.add_argument('--ae_weight_decay', type=float, default=5e-3, help="AutoEncoder Weight decay in optimizer")

	parser.add_argument('--cnn_learning_rate', type=float, default=1e-4, help="CNN Learning Rate")
	parser.add_argument('--cnn_epochs', type=int, default=150, help="CNN Num of Epochs")
	parser.add_argument('--cnn_weight_decay', type=float, default=5e-6, help="CNN Weight decay in optimizer")

	parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate(percentage of	deactivate neurons")
	parser.add_argument('--test_interval', type=int, default=5, help="Test and model save interval during train")

	parser.add_argument('--svdd_mode', type=str, default="one-class", help="SVDD mode(default: one-class) choose: one-class, soft")
	parser.add_argument('--nu', type=float, default=0.1, help="Set Nu which is used for radius")
	#Flag params
	parser.add_argument('--pretrain', type=int, default=1, help="Pretraining start Flag(Auto encoder)")
	parser.add_argument('--train', type=int, default=1, help="training start Flag")
	parser.add_argument('--test', type=int, default=1, help="Test start Flag")
	

	#Preprocessing params
	parser.add_argument('--normal_class', type=int, default=0, help="Select normal class(default: 0)")


	parser.add_argument('--log_file', type=str, default='log.txt', help="Log file name")

	args = parser.parse_args()


	#Log setting
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	log_file = './log/'+args.log_file
	file_handler = logging.FileHandler(log_file)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	#Log Info
	logger.info('Device: {}'.format(args.device))
	
	pp = Preprocessing()

	assert args.nu >=0 and args.nu <= 1
	assert args.svdd_mode in ['one-class', 'soft']

	train_data = MNIST(root='./', train=True, download=True, transform=ToTensor())
	test_data = MNIST(root='./', train=False, download=True, transform=ToTensor())

	train_data, train_label = train_data.data, train_data.targets
	train_data = pp.normalize(train_data,fit=True)
	test_data, test_label = test_data.data, test_data.targets
	test_data = pp.normalize(test_data,fit=False)

	train_dset = Dataset(train_data,train_label,normal=args.normal_class,  train=True)
	train_loader = torch.utils.data.DataLoader(dataset=train_dset, batch_size = args.batch_size, shuffle=True, drop_last=True)

	AE = MNIST_AE(rep_dim=32)

	Trainer = AE_Trainer(args)
	Trainer.set_train_loader(train_loader)

	if args.pretrain == 1:
		AE = Trainer.train(AE)

	test_dset = Dataset(test_data,test_label,normal=args.normal_class,train=False)
	test_loader = torch.utils.data.DataLoader(dataset=test_dset, batch_size = args.batch_size, shuffle=False, drop_last=True)
	Trainer.set_test_loader(test_loader)

	AE.load_state_dict(torch.load('save_model/'+str(AE.name)+'.pth'))
	"""
	# AutoEncoder Test result visualization
	Input, Output = Trainer.test(AE, test_loader)

	Input = Input.reshape((10,10,28,28))
	Input = Input.transpose((0,2,1,3))
	Input = Input.reshape((280, 280))
	
	Output = Output.reshape((10,10,28,28))
	Output = Output.transpose((0,2,1,3))
	Output = Output.reshape((280, 280))


	fig=plt.figure()

	ax1 = fig.add_subplot(2,1,1)
	ax1.set_title("Normal")
	plt.imshow(Input)

	ax2 = fig.add_subplot(2,1,2)
	ax2.set_title("Prediction")
	plt.imshow(Output)

	fig.tight_layout()

	plt.show()
	"""

	#Deep SVDD
	CNN = MNIST_CNN(rep_dim=32)
	Trainer = CNN_Trainer(args)
	Trainer.set_train_loader(train_loader)
	Trainer.set_test_loader(test_loader)
	R, C, CNN = Trainer.train(AE, CNN)

	model_dict = torch.load('./save_model/' + str(CNN.name) + '.tar')

	R = model_dict['R']
	C = model_dict['C']
	CNN.load_state_dict(model_dict['model'])
	Trainer.test(R, C, CNN)
			

