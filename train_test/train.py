import pdb
import torch
from tqdm import tqdm
from test import test_epoch
import os
import time
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, './util')
from utils import save_model

def train_model(args, model, data_loader, data_loader_valid, loss_func, optimizer, scheduler):
    """
    Trains the model and evaluates its performance.
    """

    # Create the output directory if it doesn't exist
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_mean_errors = []
    valid_mean_errors = []
    
    # Iterate over the specified number of epochs
    for epoch in tqdm(range(args.epoch_num)):
        start_time = time.time()
        print(f'Starting Epoch {epoch}')

        # Evaluate model performance
        max_mre_valid, min_mre_valid, mean_mre_valid, sigma3_valid = test_epoch(
            args=args, model=model, data_loader=data_loader_valid, loss_func=loss_func)
        print(f'Validation - Max Relative Error: {max_mre_valid}, Mean RE: {mean_mre_valid}, Min RE: {min_mre_valid}, 3σ: {sigma3_valid}')
        print(f'Last Learning Rate: {scheduler.get_last_lr()[0]}')

        max_mre_train, min_mre_train, mean_mre_train, sigma3_train = test_epoch(
            args=args, model=model, data_loader=data_loader, loss_func=loss_func)
        print(f'Training - Max Relative Error: {max_mre_train}, Mean RE: {mean_mre_train}, Min RE: {min_mre_train}, 3σ: {sigma3_train}')
        
        # Store mean errors for plotting
        train_mean_errors.append(mean_mre_train)
        valid_mean_errors.append(mean_mre_valid)

        # Check if the model meets the criteria
        # if max_mre_train < args.march_tol or mean_mre_train < args.march_tol * 0.1:
        # #    save_model(model, args, Nt, bestModel=True)
        #      scheduler.step()
        #      continue

        # Train the model for one epoch
        model = train_epoch(args=args, model=model, data_loader=data_loader, loss_func=loss_func, optimizer=optimizer)

        print(f'Epoch {epoch} elapsed time: {time.time() - start_time}s')

        # Plot and save the figure every 100 epochs
        if epoch % 100 == 0 and epoch != 0:
            plt.figure(figsize=(10, 5))
            plt.loglog(train_mean_errors, label='Training Mean RE')
            plt.loglog(valid_mean_errors, label='Validation Mean RE')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Relative Error')
            plt.title(f'Mean Relative Error per Epoch - Epoch {epoch}')
            plt.legend()

            # Save the figure
            plt.savefig(os.path.join(output_dir, f"mean_error_epoch_{epoch}.png"))
            plt.close()

    # Save the final model
    save_model(model, args, 0, bestModel=False)
    print("Model saved")



	
# Trains the model for one epoch through the entire dataset.
def train_epoch(args, model, data_loader, loss_func, optimizer):
    # Print the total number of iterations (batches) in this epoch
    print('Number of total seqs = ', len(data_loader))
    assert len(data_loader) == 1
    # Iterate over each batch in the data loader
    for batch in (data_loader):
        # Prepare batch data
        inputs = batch.to(args.device).float()

        # Predict future states
        # Torch need N, C, H, W
        for t in tqdm(range(inputs.size(1))):
            model.train()
            HDM = inputs[:, t, :, :, :]
            LDM = model.embed(HDM)
            HDM_approx = model.recover(LDM)
            loss = loss_func(HDM_approx, HDM)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model