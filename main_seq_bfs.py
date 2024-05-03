# General Package
import argparse
import sys
import pdb
from datetime import datetime
import os
import numpy as np
import json
import torch
import time
from torch.utils.data import DataLoader

# Internal package
sys.path.insert(0, './util')
from utils import save_args
from utils import read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel_copy import SequentialModel as transformer

sys.path.insert(0, './train_test_seq')
from train_seq import train_seq_shift
sys.path.insert(0, './model')
from spatial_model import CNN_encoder_decoder_2D
from train_encoder_decoder import Args

class Args_Transformer:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		"""
		for training args txt
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
		default = 'output/bfs_les_2024_01_30_15_57_29',
								help = 'load the args_train')
		self.parser.add_argument("--Nt_read",
								 default = 0,
								 help = "Which Nt model we need to read")

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
								 default=41,
								 help = 'maximum sequence length that the model is expected to handle during training, including both the input sequence and the prediction target')
		self.parser.add_argument("--start_n", 
								 default=0,
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span",
								 default=8000,
								 help='the total step of the data from the staring step')



		self.parser.add_argument("--trajec_max_len_valid", 
								 default=41,
								 help = 'max seq_length (per seq) to valid the model')
		self.parser.add_argument("--start_n_valid", 
								 default=8000,
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span_valid",
								 default=42,
								 help='the total step of the data from the staring step')
			   

		"""
		for model
		"""
		self.parser.add_argument("--n_layer", 
								 default =8,#8
								 help = 'number of attention layer')
		self.parser.add_argument("--output_hidden_states", 
								 default= True,
								 help='out put hidden matrix')
		self.parser.add_argument("--output_attentions",
								 default = True,
								 help = 'out put attention matrix')
		self.parser.add_argument("--n_ctx",
								 default = 40,
								 help='context for short, i.e. how many previous timesteps the model can consider when making a prediction')
		self.parser.add_argument("--n_embd", 
								 default = 256,
								 help='The hidden state dim transformer to predict') 
		self.parser.add_argument("--n_head", 
								 default = 4,
								 help='number of head per layer')
		self.parser.add_argument("--embd_pdrop",
								 default = 0.0,
								 help='T.B.D')
		self.parser.add_argument("--layer_norm_epsilon", 
								 default=1e-5,
								 help='############ Do not change')
		self.parser.add_argument("--attn_pdrop", 
								 default = 0.0,
								 help='T.B.D')
		self.parser.add_argument("--resid_pdrop", 
								 default = 0.0,
								 help='T.B.D')
		self.parser.add_argument("--activation_function", 
								 default = "relu",
								 help='Trust OpenAI and Nick')
		self.parser.add_argument("--initializer_range", 
								 default = 0.02,
								 help='Trust OpenAI and Nick')
		self.parser.add_argument("--unet_dim",
						   default=32,
						   help='The unet dimension')
		self.parser.add_argument("--num_sample_steps", 
								 default=20,
								 help='The noise forward/reverse step')
		
		
		"""
		for training
		"""
		self.parser.add_argument("--start_Nt",
								 default=1,
								 help='The starting length of forward propgatate')
		self.parser.add_argument("--d_Nt",
								 default=1,
								 help='The change length of forward propgatate')
		self.parser.add_argument("--batch_size",
								 default=16, #max 16->0.047
								 help = 'how many seqs you want to train together per bp')
		self.parser.add_argument("--batch_size_valid",
								 default=16, #max 16->0.047
								 help = 'how many seqs you want to train together per valid')
		self.parser.add_argument("--shuffle",
								 default=True,
								 help = 'shuffle the batch')
		self.parser.add_argument("--device",
								 default='cuda:1')
		self.parser.add_argument("--epoch_num", 
								 default = 5000000,
								 help='epoch_num')
		self.parser.add_argument("--learning_rate", 
								 default = 1e-4,
								 help='learning rate')
		self.parser.add_argument("--gamma",
								 default=0.99083194489,
								 help='learning rate decay')
		self.parser.add_argument("--march_tol", 
								 default=0.05,
								 help='march threshold for Nt + 1')
		
	def update_args(self):
		args = self.parser.parse_args()
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,'transformer_folder')
		if not os.path.isdir(args.experiment_path):
			os.makedirs(args.experiment_path)
		args.model_save_path = os.path.join(args.experiment_path,'model_save')
		if not os.path.isdir(args.model_save_path):
			os.makedirs(args.model_save_path)
		args.logging_path = os.path.join( args.experiment_path,'logging') 
		if not os.path.isdir(args.logging_path):
			os.makedirs(args.logging_path)

		args.encoder_decoder_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'logging','args.txt' )    
		return args


if __name__ == '__main__':
	args = Args_Transformer()
	args = args.update_args()
	args_train_encoder_decoder = read_args_txt(Args(), args.encoder_decoder_args_txt)
	args_train_encoder_decoder.device = args.device
	assert args.n_embd == args_train_encoder_decoder.n_embd
	save_args(args)   
	 
	"""
	pre-check
	"""
	assert args.trajec_max_len == args.n_ctx + 1
	# for each sequence of length n_ctx that the model uses as input,
	# it is expected to predict the next point in the sequence

	"""
	fetch data
	"""
	print('Start data_set')
	tic = time.time()
	data_set_train = bfs_dataset(data_location  = args.data_location,
								 trajec_max_len = args.trajec_max_len,
								 start_n        = args.start_n,
								 n_span         = args.n_span)
	data_set_test_on_train = bfs_dataset(data_location  = args.data_location,
										 trajec_max_len = args.trajec_max_len_valid,
										 start_n        = args.start_n,
										 n_span         = args.n_span)
	data_set_valid = bfs_dataset(data_location  = args.data_location,
								 trajec_max_len = args.trajec_max_len_valid,
								 start_n        = args.start_n_valid,
								 n_span         = args.n_span_valid)
	data_loader_train = DataLoader(dataset    = data_set_train,
								   shuffle    = args.shuffle,
								   batch_size = args.batch_size)
	data_loader_test_on_train = DataLoader(dataset    = data_set_test_on_train,
										   shuffle    = args.shuffle,
										   batch_size = args.batch_size_valid)
	data_loader_valid = DataLoader(dataset    = data_set_valid,
								   shuffle    = args.shuffle,
								   batch_size = args.batch_size_valid)
	print('Done data-set use time ', time.time() - tic)
	"""
	create model
	"""
	model = transformer(args).to(args.device).float()
	# model_bug = transformer_bug(args).to(args.device).float()
	# pdb.set_trace()
	# xn = torch.randn(1, 1, 256).to(args.device).float()
	print('Number of parameters: {}'.format(model._num_parameters()))
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
	creat and load encoder_decoder
	"""
	encoder_decoder = CNN_encoder_decoder_2D(
	n_embd            =args_train_encoder_decoder.n_embd,
	layer_norm_epsilon=args_train_encoder_decoder.layer_norm_epsilon,
	embd_pdrop        =args_train_encoder_decoder.embd_pdrop,
	n_channels        =args_train_encoder_decoder.n_channels
	).to(args.device).float()
	
	print('Number of parameters for encoder-decoder: {}'.format(encoder_decoder._num_parameters()))

	encoder_decoder.load_state_dict(torch.load(args_train_encoder_decoder.current_model_save_path+'model_epoch_'+str(args.Nt_read),
											   map_location=torch.device(args.device)))


	"""
	train
	"""
	train_seq_shift(args=args, 
					model=model, 
					data_loader=data_loader_train, 
					data_loader_copy = data_loader_test_on_train,
					data_loader_valid = data_loader_valid,
					loss_func=loss_func, 
					optimizer=optimizer,
					scheduler=scheduler,
					encoder_decoder = encoder_decoder)