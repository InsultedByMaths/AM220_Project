# General Package
import argparse  # Library for parsing command line arguments
import sys
import pdb  # Python Debugger for interactive debugging
from datetime import datetime  # Library to work with dates and times
import os  # Library for interacting with the operating system
import numpy as np  # Numerical computing library
import json  # Library for JSON manipulation
import torch  # PyTorch library for tensor computations
from torch.utils.data import DataLoader  # PyTorch utility for data loading

# Internal package
sys.path.insert(0, './util')  # Adding 'util' directory to system path for importing modules
from utils import save_args  # Importing 'save_args' function from 'utils' module

sys.path.insert(0, './data')  # Adding 'data' directory to system path for importing modules
from data_bfs_preprocess import bfs_dataset  # Importing 'bfs_dataset' class for BFS data handling

sys.path.insert(0, './model')
from spatial_model import CNN_encoder_decoder_2D

#sys.path.insert(0, './transformer')
#from sequentialModel import SequentialModel as transformer

sys.path.insert(0, './train_test')
from train import train_model

import time  # Library for time-related functions

# Argument parser class for handling command line arguments
class Args:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for dataset
		"""
		self.parser.add_argument("--dataset",
								 default='bfs_les',
								 help='name it')
		self.parser.add_argument("--data_location", 
								 default = ['./data/data0.npy',
											'./data/data1.npy'],
								 help='the relative or abosolute data.npy file')
		self.parser.add_argument("--trajec_max_len", 
								 default=3000,
								 help = 'max seq_length (per seq) to train the model')
		self.parser.add_argument("--start_n", 
								 default=200,
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span",
								 default=3001,
								 help='the total step of the data from the staring step')



		self.parser.add_argument("--trajec_max_len_valid", 
								 default=200,
								 help = 'max seq_length (per seq) to valid the model')
		self.parser.add_argument("--start_n_valid", 
								 default=3200,
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span_valid",
								 default=201,
								 help='the total step of the data from the staring step')
			   

		"""
		for model
		"""
		self.parser.add_argument("--n_embd", 
								 default = 256,
								 help='The hidden state dim fed into mlp to further reduce dim,\
								       must be multiples of 256') 
		self.parser.add_argument("--embd_pdrop",
								 default = 0.0,
								 help='T.B.D')
		self.parser.add_argument("--layer_norm_epsilon", 
								 default=1e-5,
								 help='############ Do not change')
		self.parser.add_argument("--n_channels", 
								 default = 2,
								 help='We only predict u and v')
		
		
		"""
		for training encoder decoder
		"""
		self.parser.add_argument("--batch_size",
								 default=1, 
								 help = 'should be set to 1 since we only train decoder and encoder')
		self.parser.add_argument("--batch_size_valid",
								 default=1, 
								 help = 'should be set to 1 since we only train decoder and encoder')
		self.parser.add_argument("--device",
								 default='cuda:0')
		self.parser.add_argument("--epoch_num", 
                         		 default=1000,
                        	     type=int,
                                 help="The number of training epochs")
		self.parser.add_argument("--learning_rate", 
								 default = 1e-5,
								 help='learning rate')
		self.parser.add_argument("--gamma",
								 default=0.99083194489,
								 help='learning rate decay')
		self.parser.add_argument("--march_tol", 
								 default=0.01,
								 help='march threshold for LR decay')

	# Method to update arguments with additional information and create necessary directories	
	def update_args(self):
		# Parsing the arguments from the command line
		args = self.parser.parse_args()

		# Adding additional attributes for output file management
        # Creating a timestamp for file naming
		args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

		# Setting up directories for outputs, logs, and model saving
		args.dir_output = 'output/'
		args.fname = args.dataset + '_' +args.time
		args.experiment_path = args.dir_output + args.fname
		args.model_save_path = args.experiment_path + '/' + 'model_save/'
		args.logging_path = args.experiment_path + '/' + 'logging/'
		args.current_model_save_path = args.model_save_path
		args.logging_epoch_path = args.logging_path + 'epoch_history.csv'

		# Creating directories if they do not exist
		if not os.path.isdir(args.logging_path):
			os.makedirs(args.logging_path)
		if not os.path.isdir(args.model_save_path):
			os.makedirs(args.model_save_path)
		
		# Returning the parsed and updated arguments
		return args

















# Main execution block
if __name__ == '__main__':
	args = Args()
	args = args.update_args()
	save_args(args) # Saving the command line arguments

	"""
	pre-check
	"""
	#assert args.coarse_dim[0]*args.coarse_dim[1]*2 == args.n_embd
	#assert args.trajec_max_len_valid == args.n_ctx + 1
	
	# Data fetching and DataLoader setup
	print('Start data_set')
	tic = time.time() # Starting a timer

	# Initialize datasets and data loaders for training, testing on training data, and validation

	# Create training dataset
	data_set_train = bfs_dataset(
		data_location=args.data_location,       # Location of the training data
		trajec_max_len=args.trajec_max_len,     # Maximum length of trajectories in the training data
		start_n=args.start_n,                   # Starting index for data selection in the training set
		n_span=args.n_span                      # Span parameter for the training data
	)


	# Create validation dataset
	data_set_valid = bfs_dataset(
		data_location=args.data_location,            # Location of the validation data
		trajec_max_len=args.trajec_max_len_valid,    # Maximum length of trajectories in the validation data
		start_n=args.start_n_valid,                  # Starting index for data selection in the validation set
		n_span=args.n_span_valid                     # Span parameter for the validation data
	)

	# DataLoader for the training dataset
	data_loader_train = DataLoader(
		dataset=data_set_train,
		shuffle=False,                  # Whether to shuffle the data each epoch
		batch_size=args.batch_size             # Batch size for training
	)

	# DataLoader for the validation dataset
	data_loader_valid = DataLoader(
		dataset=data_set_valid,
		shuffle=False,                  # Whether to shuffle the data each epoch
		batch_size=args.batch_size_valid       # Batch size for validation
	)

	print('Done data-set use time ', time.time() - tic) # Printing elapsed time
	
	"""
	create model
	"""
	#model = transformer(args).to(args.device).float()
	#print('Number of parameters: {}'.format(model._num_parameters()))

	model = CNN_encoder_decoder_2D(n_embd = args.n_embd,
								   layer_norm_epsilon = args.layer_norm_epsilon,
								   embd_pdrop = args.embd_pdrop,
								   n_channels = args.n_channels).to(args.device).float()
	
	print('Number of parameters: {}'.format(model._num_parameters()))
	'''
	# Create mock inputs with specific dimensions
	H, W = 512, 512  # Example height, and width
	x_mock = torch.zeros([args.batch_size, args.n_channels, H, W])  # Replace with random or zeros

	print("Data dimensions before encoding:", x_mock.shape)

    # Call the embed function
	g = model.embed(x_mock)

    # Print the output dimensions
	print("Output dimensions after encoding:", g.shape)

    # Call the 'recover' function
	x_recovered = model.recover(g)

    # Print the output dimensions
	print("Output dimensions of recover:", x_recovered.shape)
	# pdb.set_trace()
	'''
	
	"""
	create loss function
	"""
	loss_func = torch.nn.MSELoss()
	
	"""
	create optimizer
	"""
	optimizer = torch.optim.Adam(model.parameters(), 
	                            lr=args.learning_rate)
	"""
	create scheduler
	"""
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
	                                            step_size=1,
	                                            gamma=args.gamma)    
	"""
	train
	"""
	train_model(args=args, 
	            model=model, 
	            data_loader=data_loader_train, 
	            data_loader_valid = data_loader_valid,
	            loss_func=loss_func, 
	            optimizer=optimizer,
	            scheduler=scheduler)