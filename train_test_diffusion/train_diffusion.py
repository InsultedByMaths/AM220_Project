from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
sys.path.insert(0, './util')
from utils import save_loss

# function for training the diffusion model.
def train_diffusion(diff_args, 
					seq_args, 
					endecoder_args,
					trainer, 
					encoder_decoder, 
					transformer,
					data_loader):
	loss_list = []  # Initialize a list to store loss values.
	
	# Loop through each epoch.
	for epoch in range(diff_args.epoch_num):
		
		# Train the model for one epoch and receive the updated model and the loss.
		model, loss = train_epoch(diff_args, 
								  seq_args, 
								  endecoder_args,
								  trainer, 
								  encoder_decoder, 
								  transformer,
								  data_loader)
		# Every epoch (excluding the first), save the current loss and model.
		if epoch % 1 == 0 and epoch > 0:
			save_loss(diff_args, loss_list + [loss], epoch)  # Save the loss to a file.
			# Save the current model specifying the epoch number in the filename.
			#model.save(path=os.path.join(diff_args.model_save_path, 'model_epoch_' + str(epoch)))
		
		# Starting from the second epoch, check if the current loss is the lowest.
		if epoch >= 1:
			if loss < min(loss_list):  # If the current loss is the lowest so far,
				# Save the loss indicating it's the best so far.
				save_loss(diff_args, loss_list + [loss], epoch)
				# Save the current model as the best model so far.
				model.save(path=os.path.join(diff_args.model_save_path, 'best_model_sofar'))
				# Save the epoch number of the best model so far to a file.
				np.savetxt(os.path.join(diff_args.model_save_path, 'best_model_sofar_epoch'), np.ones(2) * epoch)
		
		# Append the current loss to the list of losses.
		loss_list.append(loss)
		
		# Print a message indicating the completion of the current epoch.
		print("finish training epoch {}".format(epoch))


# function to train the model for a single epoch.
def train_epoch(diff_args, 
				seq_args, 
				endecoder_args,
				trainer, 
				encoder_decoder, 
				transformer,
				data_loader):
	loss_epoch = []  # Initialize a list to store loss values for the epoch.

	# Print the total number of iterations (batches) in this epoch.
	print('Iteration is ', len(data_loader))

	# Loop through each batch in the data loader.
	for iteration, batch in (enumerate(data_loader)):
		if iteration > 2:
			break
		# Move the batch to the specified device and convert it to float.
		batch = batch.to(diff_args.device).float()

		# Extract batch size and time dimension.
		bsize = batch.shape[0]  # Batch size.
		ntime = batch.shape[1]  # Time dimension.
		# Down-sample the batch and then up-sample it.
		# Reshape for down-sampling to process each time slice as a separate image.
		with torch.no_grad():
			embedded_data = encoder_decoder.embed(batch.view(bsize * ntime, *batch.shape[2:]))
			embedded_data = embedded_data.view(bsize, ntime, -1)

		with torch.no_grad():
			context_indices = [i for i in range(1, seq_args.n_ctx)]
			past_state = None
			initial_input = embedded_data[:, 0:1, :]
			memory_output = []
			for step in range(ntime-1):
				if step == 0 or (past_state[0][0].shape[2] < seq_args.n_ctx and step > 0):
					# pdb.set_trace()
					prediction, past_state, _, _ = transformer(inputs_embeds=initial_input, past=past_state)
				else:
					past_state = [[past_state[layer][0][:, :, context_indices, :], past_state[layer][1][:, :, context_indices, :]] for layer in range(seq_args.n_layer)]
					prediction, past_state, _, _ = transformer(inputs_embeds=initial_input, past=past_state)
				initial_input = prediction
				memory_output.append(prediction)

			concatenated_predictions = torch.cat(memory_output, dim=1)
		
		
		embedded_data_reshape = \
			embedded_data[0,1:].reshape(
		   [ntime-1,
			diff_args.DiffImageHeight,
			int(endecoder_args.n_embd/diff_args.DiffImageHeight)])
		concatenated_predictions_reshape = \
		concatenated_predictions[0].reshape(
		   [ntime-1,
			diff_args.DiffImageHeight,
			int(endecoder_args.n_embd/diff_args.DiffImageHeight)])
		for i in tqdm(range(ntime-1-diff_args.D_ln)):
			off_manifold =  \
			concatenated_predictions_reshape[i:i+diff_args.D_ln].unsqueeze(0).unsqueeze(0)
			on_manifold = \
			embedded_data_reshape[i:i+diff_args.D_ln].unsqueeze(0).unsqueeze(0)		
			loss = trainer(on_manifold, cond_images=off_manifold, unet_number=1, ignore_time=False)
			# Update the trainer's parameters (e.g., optimizer parameters) after calculating the loss.
			trainer.update(unet_number=1)
			loss_epoch.append(loss)
	
	# Return the trainer and the average loss for the epoch.
	return trainer, sum(loss_epoch) / len(loss_epoch)
