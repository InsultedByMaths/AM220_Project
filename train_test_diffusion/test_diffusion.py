from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
import pickle

def test_diffusion(args_latentdiff_eval,
				   args_encoder_decoder,
				   args_seq,
				   args_latentdiff,
				   encoder_decoder,
				   transformer,
				   trainer,
				   data_loader):
	try:
		os.makedirs(args_latentdiff_eval.experiment_path+'/contour')
	except:
		pass
	contour_dir = args_latentdiff_eval.experiment_path+'/contour'
	loss_func = torch.nn.MSELoss()
	Nt = args_latentdiff_eval.test_Nt
	
	with torch.no_grad():
		IDHistory = [i for i in range(1, args_seq.n_ctx)]
		REs = []
		REs_fine = []
		print('total ite', len(data_loader))
		for iteration, batch in tqdm(enumerate(data_loader)):
			batch = batch.to(args_latentdiff_eval.device).float()
			b_size = batch.shape[0]
			assert b_size == 1
			num_time = batch.shape[1]
			num_velocity = 2
			batch_embedded = encoder_decoder.embed(batch.view(b_size * num_time, *batch.shape[2:]))
			# batch_embedded = [batch_size * num_time, n_embd]
			batch_embedded = batch_embedded.view(b_size, num_time, -1)  # Reshape to reintroduce the time dimension
			# batch_embedded = [batch_size, time, n_embd]
			

			"""
			without latent diff correction
			"""
			past = None
			xn = batch_embedded[:,0:1,:]
			previous_len = 1 
			mem = []
			for j in (range(Nt)):
				if j == 0:
					xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
				elif past[0][0].shape[2] < args_seq.n_ctx and j > 0:
					xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
				else:
					past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args_seq.n_layer)]
					xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
				xn = xnp1
				mem.append(xn)
			mem=torch.cat(mem,dim=1)

			"""
			with latent diff correction
			"""
			past = None
			xn = batch_embedded[:,0:1,:]
			previous_len = 1 
			mem_corected = []
			for i in range(Nt // args_latentdiff_eval.latent_interval):
				# forward
				for j in (range(args_latentdiff_eval.latent_interval)):
					if j == 0 or (past[0][0].shape[2] < args_seq.n_ctx and j > 0):
						xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
					else:
						past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args_seq.n_layer)]
						xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
					xn = xnp1
					mem_corected.append(xn)
				# apply correction
				mem_tobecorrected = mem_corected[-args_latentdiff.D_ln:]
				mem_tobecorrected = torch.cat(mem_tobecorrected, dim=1)
				mem_tobecorrected = mem_tobecorrected.reshape([mem_tobecorrected.shape[0],
												               mem_tobecorrected.shape[1],
															   args_latentdiff.DiffImageHeight,
															   int(args_encoder_decoder.n_embd/args_latentdiff.DiffImageHeight)])
				mem_tobecorrected = mem_tobecorrected.unsqueeze(0)
				mem_local_corrected = trainer.sample(video_frames = args_latentdiff.D_ln,
				                                     cond_images  = mem_tobecorrected)
				
				mem_corected[-args_latentdiff.D_ln:] = \
				[mem_local_corrected[0,:,k,:,:].reshape([1,1,-1]) for k in range(args_latentdiff.D_ln)]
				xn = mem_corected[-1]
				past = None

			if len(mem_corected)< Nt:
				for j in range(Nt - len(mem_corected)):
					if j == 0 or (past[0][0].shape[2] < args_seq.n_ctx and j > 0):
						xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
					else:
						past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args_seq.n_layer)]
						xnp1,past,_,_=transformer(inputs_embeds = xn, past=past)
					xn = xnp1
					mem_corected.append(xn)
			mem_corected=torch.cat(mem_corected,dim=1)
			


			local_batch_size = mem.shape[0]
			for i in tqdm(range(local_batch_size)):
				prediction = mem[i:i+1]
				prediction_corrected = mem_corected[i:i+1]
				truth      = batch_embedded[i:i+1,previous_len:previous_len+Nt,:]

				assert prediction.shape[0] == truth.shape[0] == 1
				bsize_here = 1
				ntime      = prediction.shape[1]

				prediction_micro_corrected = \
				encoder_decoder.recover(prediction_corrected[0]).reshape([bsize_here, ntime, num_velocity, 512, 512])
				prediction_micro = encoder_decoder.recover(prediction[0]).reshape([bsize_here, ntime, num_velocity, 512, 512])
				recover_micro      = encoder_decoder.recover(truth[0]).reshape([bsize_here, ntime, num_velocity, 512, 512])
				

				
				prediction_micro = prediction_micro.permute([0,2,1,3,4])
				prediction_micro_corrected = \
				prediction_micro_corrected.permute([0,2,1,3,4])
				recover_micro    = recover_micro.permute([0,2,1,3,4])
				truth_micro      = batch.permute([0,2,1,3,4])[:,:,1:]
				truth_macro      = batch_embedded[:,1:,:]
				prediction_macro = mem
				prediction_macro_corrected = mem_corected


				
				
				
				
				seq_name = 'batch'+str(iteration)+'sample'+str(i)
				try:
					os.makedirs(contour_dir+'/'+seq_name)
				except:
					pass

				DIC = {"prediction_micro":prediction_micro.detach().cpu().numpy(),
		   			   "prediction_micro_corrected":prediction_micro_corrected.detach().cpu().numpy(),
					   "recon_micro":recover_micro.detach().cpu().numpy(),
					   "truth_micro":truth_micro.detach().cpu().numpy(),
					   "truth_macro":truth_macro.detach().cpu().numpy(),
					   "prediction_macro":prediction_macro.detach().cpu().numpy(),
					   "prediction_macro_corrected":prediction_macro_corrected.detach().cpu().numpy()}
				pickle.dump(DIC, open(contour_dir+'/'+seq_name+"/DIC.npy", 'wb'), protocol=4)
				
	