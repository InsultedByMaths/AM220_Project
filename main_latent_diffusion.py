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
from utils import read_args_txt, save_args
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel_copy import SequentialModel as Transformer
sys.path.insert(0, './model')
from spatial_model import CNN_encoder_decoder_2D
sys.path.insert(0, './train_test_diffusion')
from  train_diffusion import train_diffusion

from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer


from train_encoder_decoder import Args as Args_encoder_decoder
from main_seq_bfs import Args_Transformer



class Args_latent_diffusion:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		"""
		for training args txt
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
								 default = 'output/bfs_les_2024_01_30_15_57_29',
								 help = 'load the args of training en/decoder and transformer')
		self.parser.add_argument("--Nt_endecoder",
								 default = 0,
								 help = "Which encoder_decoder model we need to read, we only save Nt = 0 at the end")
		self.parser.add_argument("--Nt_transformer",
								 default = 36,
								 help = 'Which transformer model we need to read')
	
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
								 default=200,
								 help = 'maximum sequence length that the model is expected to handle during training, including both the input sequence and the prediction target')
		"""
		for model of latent diffusion
		"""
		self.parser.add_argument("--DiffImageHeight",
						   		 default = 16,
								 help    = 'Diffusion height')
		self.parser.add_argument("--D_ln",
						   		 default = 10,
								 help = 'Diameter of latent neighborhood')
		self.parser.add_argument("--unet_dim",
								  default = 32,
								  help = 'The unet dimension')
		self.parser.add_argument("--num_sample_steps",
								 default=20,
								 help='The noise forward/inverse step')
		
		"""
		for training
		"""
		self.parser.add_argument("--epoch_num",
								 default = 800)
		self.parser.add_argument("--device",
								default="cuda:1")
		

		
	def update_args(self):
		args = self.parser.parse_args()
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,'latent_diff_folder')
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
		args.transformer_args_txt = os.path.join(args.bfs_dynamic_folder,
											   		'transformer_folder',
													'logging','args.txt' )
			
		return args


if __name__ == '__main__':
	"""
	Diff args
	"""
	diff_latent_args = Args_latent_diffusion()
	diff_latent_args = diff_latent_args.update_args()
	save_args(diff_latent_args)

	"""
	Transformer & ende-coder args
	"""
	args_seq = read_args_txt(Args_Transformer(), 
							 diff_latent_args.transformer_args_txt)
	args_endecoder = read_args_txt(Args_encoder_decoder(), 
								   diff_latent_args.encoder_decoder_args_txt)
	
	"""
	Fetch dataset
	"""
	data_set = bfs_dataset(data_location  = args_seq.data_location,
						   trajec_max_len = diff_latent_args.trajec_max_len, #seq_args.trajec_max_len,
						   start_n        = args_seq.start_n,
						   n_span         = args_seq.n_span)
	data_loader = DataLoader(dataset=data_set, 
							 shuffle=True,
							 batch_size=1)
	
	"""
	Load encoder-decoder
	"""
	encoder_decoder = CNN_encoder_decoder_2D(n_embd = args_endecoder.n_embd,
											 layer_norm_epsilon = args_endecoder.layer_norm_epsilon,
											 embd_pdrop = args_endecoder.embd_pdrop,
											 n_channels = args_endecoder.n_channels
											 ).to(diff_latent_args.device).float()
	print('Number of encoder decoder parameters: {}'.format(encoder_decoder._num_parameters()))
	encoder_decoder.load_state_dict(torch.load(args_endecoder.current_model_save_path+'model_epoch_'+str(diff_latent_args.Nt_endecoder),map_location=torch.device(diff_latent_args.device)))

	"""
	Load Transformer
	"""	
	transformer = Transformer(args_seq).to(diff_latent_args.device).float()
	print('Number of parameters: {}'.format(transformer._num_parameters()))
	transformer.load_state_dict(torch.load(args_seq.model_save_path+'/model_epoch_'+str(diff_latent_args.Nt_transformer),map_location=torch.device(diff_latent_args.device)))
	
	"""
	Create diffusion model
	"""
	unet1 = Unet3D(dim = diff_latent_args.unet_dim,
				   cond_images_channels = 1,
				   memory_efficient=True,
				   dim_mults=(1,2,4,8)).to(torch.device(diff_latent_args.device))
	image_sizes = (diff_latent_args.DiffImageHeight)
	image_width = (int(args_endecoder.n_embd/diff_latent_args.DiffImageHeight))
	imagen = ElucidatedImagen(
		unets = (unet1),
		image_sizes = image_sizes,
		image_width = image_width,
		channels = 1,   # Hard-code because of latent space 
		random_crop_sizes = None,
		num_sample_steps = diff_latent_args.num_sample_steps, # original is 10
		cond_drop_prob = 0.1,
		sigma_min = 0.002,
		sigma_max = (80),      # max noise level, double the max noise level for upsampler  （80，160）
		sigma_data = 0.5,      # standard deviation of data distribution
		rho = 7,               # controls the sampling schedule
		P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
		P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
		S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
		S_tmin = 0.05,
		S_tmax = 50,
		S_noise = 1.003,
		condition_on_text = False,
		auto_normalize_img = False  # Han Gao make it false
		).to(torch.device(diff_latent_args.device))
	trainer = ImagenTrainer(imagen, device =torch.device(diff_latent_args.device))
	train_diffusion(diff_args = diff_latent_args,
               		seq_args = args_seq,
					endecoder_args = args_endecoder,
               		trainer = trainer,
					encoder_decoder = encoder_decoder,
					transformer = transformer,
	                data_loader = data_loader)	





	





