import argparse
import pdb
import os
import sys
from torch.utils.data import DataLoader
import torch

# from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from main_seq_bfs import Args as Args_seq
from train_encoder_decoder import Args as Args_encoder_decoder
sys.path.insert(0, './util')
from utils import read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel_copy import SequentialModel as Transformer
sys.path.insert(0, './train_test_seq')
from test_seq import test_final_overall 
sys.path.insert(0, './model')
from spatial_model import CNN_encoder_decoder_2D

class Args_final_eval:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for finding the dynamics dir
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
								 default='output/bfs_les_2024_01_30_15_57_29',
								 help='all the information of bfs training')
		
		"""
		reading encoder_decoder
		"""
		self.parser.add_argument("--Nt_read",
								 default = 0,
								 help = "Which encoder_decoder we need to read")
		"""
		reading the transformer
		"""
		self.parser.add_argument("--load_epoch",
                         		default=36,
                         		type=int,
                         		help="Specify the epoch (actually is Nt) number from which to load the transformer model.")


		"""
		for dataset
		"""
		self.parser.add_argument("--trajec_max_len", 
								 default=80,
								 help = 'max seq_length (per seq) to test the model')
		self.parser.add_argument("--start_n", 
								 default=9500,
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span",
								 default=81,
								 help='the total step of the data from the staring step')

		"""
		for seq_net_eval
		"""
		self.parser.add_argument("--test_Nt", 
								 default=79,
								 help = 'How many step you want to proceed!')
		


		
		"""
		for eval dataset hyperparameter
		"""
		self.parser.add_argument("--batch_size",
						         help='make it 1 for now',
								 default = 1)
		self.parser.add_argument("--device", type=str, default = "cuda:0")
		


	def update_args(self):
		args = self.parser.parse_args()
		args.encoder_decoder_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'logging','args.txt' )
		args.seq_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'transformer_folder',
										 'logging','args.txt')
		
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,
											'transformer_folder',
											'experiment_transformer')
		if not os.path.isdir(args.experiment_path):
			os.makedirs(args.experiment_path)
		return args

if __name__ == '__main__':
	"""
	Fetch args
	"""
	args_final = Args_final_eval()
	args_final = args_final.update_args()	
	args_seq = read_args_txt(Args_seq(), 
							  args_final.seq_args_txt)
	args_encoder_decoder = read_args_txt(Args_encoder_decoder(), 
							  args_final.encoder_decoder_args_txt)
	
	"""
	Fetch dataset
	"""
	data_set = bfs_dataset(data_location  = args_seq.data_location,
						   trajec_max_len = args_final.trajec_max_len,
						   start_n        = args_final.start_n,
						   n_span         = args_final.n_span)
	
	data_loader = DataLoader(dataset=data_set, 
							 shuffle=False,
							 batch_size=args_final.batch_size)

	"""
	Fetch models
	"""
	encoder_decoder = CNN_encoder_decoder_2D(
	n_embd            =args_encoder_decoder.n_embd,
	layer_norm_epsilon=args_encoder_decoder.layer_norm_epsilon,
	embd_pdrop        =args_encoder_decoder.embd_pdrop,
	n_channels        =args_encoder_decoder.n_channels
	).to(args_final.device).float()
	# model = transformer(args_seq).to(args_final.device).float()
	print('Number of encoder decoder parameters: {}'.format(encoder_decoder._num_parameters()))
	encoder_decoder.load_state_dict(torch.load(args_encoder_decoder.current_model_save_path+'model_epoch_'+str(args_final.Nt_read),map_location=torch.device(args_final.device)))	
	
	transformer = Transformer(args_seq).to(args_final.device).float()
	print('Number of parameters: {}'.format(transformer._num_parameters()))
	transformer.load_state_dict(torch.load(args_seq.model_save_path+'/model_epoch_'+str(args_final.load_epoch),map_location=torch.device(args_final.device)))
	test_final_overall(args_final, 
					   args_encoder_decoder, 
					   args_seq, 
					   encoder_decoder, 
					   transformer,
					   data_loader)