import pdb
import torch
from tqdm import tqdm
import os
import time
import sys

from test_seq import test_epoch

sys.path.insert(0, './util')
from utils import save_model

def train_seq_shift(args, 
					model, 
					data_loader, 
					data_loader_copy,
					data_loader_valid,
					loss_func, 
					optimizer,
					scheduler,
					encoder_decoder):
	# N C H W
	

	Nt = args.start_Nt
	for epoch in tqdm(range(args.epoch_num)):
		tic = time.time()
		print('Start epoch '+ str(epoch)+' at Nt ', Nt)
		if epoch >-1:
			max_mre,min_mre, mean_mre, sigma3 = test_epoch(args=args,
														   model=model, 
														   data_loader=data_loader_valid,
														   loss_func=loss_func,
														   num_predict_steps = Nt,
														   encoder_decoder=encoder_decoder,
														   iteration_threshold = 2)
			print('#### max  re valid####=',max_mre)
			print('#### mean re valid####=',mean_mre)
			print('#### min  re valid####=',min_mre)
			print('#### 3 sigma valid####=',sigma3)
			print('Last LR is '+str(scheduler.get_last_lr()))
			max_mre,min_mre, mean_mre, sigma3 = test_epoch(args = args,
														   model = model, 
														   data_loader = data_loader_copy,
														   loss_func = loss_func,
														   num_predict_steps = Nt,
														   encoder_decoder=encoder_decoder,
														   iteration_threshold = 5)
			print('#### max  re train####=',max_mre)
			print('#### mean re train####=',mean_mre)
			print('#### min  re train####=',min_mre)
			print('#### 3 sigma train ####=',sigma3)
			if (max_mre < args.march_tol) or (mean_mre < args.march_tol*0.1):
				save_model(model, args, Nt, bestModel = False)
				Nt += args.d_Nt
				scheduler.step()
				continue
		
		model = train_epoch(args=args,
							model=model, 
							data_loader=data_loader,
							loss_func=loss_func,
							optimizer=optimizer,
							encoder_decoder=encoder_decoder)
				
		print('Epoch elapsed ', time.time()-tic)

def train_epoch(args, model, data_loader, loss_func, optimizer, encoder_decoder):
	"""
	Trains a Transformer model for one epoch using an encoder-decoder model for data preprocessing.

	Args:
		args: configuration object containing parameters like device, coarse_dim, and n_ctx.
		model: The Transformer model to be trained.
		data_loader: DataLoader for iterating over the dataset.
		loss_func: The loss function used for training.
		optimizer: The optimizer used for updating model parameters.
		encoder_decoder: The encoder-decoder model used for embedding the input data.
	"""
	print('Nit = ', len(data_loader))  # Print the number of iterations (batches) in this epoch

	# Iterate over each batch in the data loader
	for iteration, batch in tqdm(enumerate(data_loader)):
		batch = batch.to(args.device).float()  # Move batch to specified device and ensure it is in float format
		# batch = [batch_size, time, channel, height, width]
		# Unpack batch dimensions
		b_size = batch.shape[0]  # Batch size
		num_time = batch.shape[1]  # Number of time steps in the batch

		# Use the encoder part of the encoder-decoder model to embed the input data
		# The encoder takes input of shape [batch_size, channels, height, width] and outputs [batch_size, n_embed]
		batch_embedded = encoder_decoder.embed(batch.view(b_size * num_time, *batch.shape[2:]))
		# batch_embedded = [batch_size * num_time, n_embd]
		batch_embedded = batch_embedded.view(b_size, num_time, -1)  # Reshape to reintroduce the time dimension
		# batch_embedded = [batch_size, time, n_embd]
		# Ensure the number of time steps is correct for the Transformer context size
		assert num_time == args.n_ctx + 1

		# Iterate over each time step in the sequence for training
		for j in range(num_time - args.n_ctx):
			model.train()  # Set model to training mode
			optimizer.zero_grad()  # Reset gradients

			# Select the input sequence for the current context
			xn = batch_embedded[:, j:j+args.n_ctx, :]

			# Forward pass: Compute the model's prediction for the next time step
			xnp1, _, _, _ = model(inputs_embeds=xn, past=None)

			# Select the target sequence for the current context
			xn_label = batch_embedded[:, j+1:j+1+args.n_ctx, :]

			# Calculate loss between the model's prediction and the target
			loss = loss_func(xnp1, xn_label)
			loss.backward()  # Backpropagation
			optimizer.step()  # Update model parameters

	return model  # Return the trained model