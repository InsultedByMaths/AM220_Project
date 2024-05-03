# Write a 2D CNN encoder decoder here

# refer Nicholas's paper https://github.com/zabaras/transformer-physx/blob/main/trphysx/embedding/embedding_cylinder.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import sys
import pdb
import sys

# This adds the parent directory ('case0') to sys.path
sys.path.insert(0, '..')

# Now import from train_encoder_decoder and util
# from train_encoder_decoder import Args
# from util.utils import save_args

# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

class CNN_encoder_decoder_2D(nn.Module):  # Define a class inheriting from PyTorch's nn.Module
	model_name = "CNN_encoder_decoder_2D"  # Class variable to store the model's name

	def __init__(self, n_embd, layer_norm_epsilon, embd_pdrop, n_channels) -> None:
		"""
		Constructor for the CNN_encoder_decoder_2D class.

		:param args: Arguments to configure the model. Expected to be an instance of a custom Args class.
		"""
		super().__init__()  # Initialize the base class (nn.Module)
		self.n_embd = n_embd
		self.n_channels = n_channels

		# Create a 2D grid of coordinates using numpy's meshgrid and linspace
		# linspace creates evenly spaced numbers over a specified interval
		# meshgrid generates coordinate matrices from coordinate vectors
		X, Y = np.meshgrid(np.linspace(-2, 14, 128), np.linspace(-4, 4, 64))

		# Create a mask based on the condition: sqrt(X^2 + Y^2) < 1
		# This mask will be True where the condition is met, and False otherwise
		# The mask is converted to a PyTorch tensor of boolean type
		self.mask = torch.tensor(np.sqrt(X**2 + Y**2) < 1, dtype=torch.bool)

		# Encoder convolutional network
		self.observableNet = nn.Sequential(
			# First convolutional layer with 4 input channels and 16 output channels.
			# Kernel size is 3x3, stride is 2 (downsampling by 2), and padding is 1.
			# 'replicate' padding mode replicates the border pixels.
			nn.Conv2d(self.n_channels, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
			# nn.BatchNorm2d(16),
			# ReLU activation function introduces non-linearity, allowing for more complex mappings.
			nn.ReLU(True),
			
			# Second convolutional layer increasing channels from 16 to 32.
			nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
			# nn.BatchNorm2d(32),
			nn.ReLU(True),

			# Third convolutional layer increasing channels from 32 to 64.
			nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
			# nn.BatchNorm2d(64),
			nn.ReLU(True),

			# Fourth convolutional layer increasing channels from 64 to 128.
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
			# nn.BatchNorm2d(128),
			nn.ReLU(True),

			# convolutional layer, output channels are 256.
			# Stride is 1, maintaining the spatial dimensions of the input.
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
			nn.ReLU(True),

			nn.Conv2d(256, self.n_embd // 256, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
			
			# Notes (original comments):
			# 8, 32, 64
			# 16, 16, 32
			# 16, 8, 16
			# 16, 4, 8
		)

		# Definition of a fully connected layer sequence for the network
		self.observableNetFC = nn.Sequential(
			# Layer normalization layer. It normalizes the input across the features.
			# 'args.n_embd' specifies the input size for normalization.
			# 'args.layer_norm_epsilon' is a small value for numerical stability.
			nn.LayerNorm(self.n_embd, eps=layer_norm_epsilon),

			# nn.Linear(config.n_embd // 32 * 4 * 8, config.n_embd-1),
			# nn.BatchNorm1d(config.n_embd, eps=config.layer_norm_epsilon),

			# Dropout layer to prevent overfitting. It randomly sets a fraction of input
			# units to 0 at each update during training, with the fraction being 'args.embd_pdrop'.
			nn.Dropout(embd_pdrop)
		)

		# Decoder convolutional network
		self.recoveryNet = nn.Sequential(
			# Upsampling layer to increase the spatial dimensions of the input feature map.
			# Uses bilinear interpolation for upsampling with scale factor 2.
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

			# Convolutional layer to process the upsampled feature map.
			# Reduces the number of channels from (args.n_embd // 32) to 128.
			# Uses a 3x3 kernel, stride of 1, and padding of 1 to maintain the spatial size.
			nn.Conv2d(self.n_embd // 256, 256, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
			nn.ReLU(),  # ReLU activation function adds non-linearity to the model.

			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
			nn.ReLU(),

			# Upsample again, followed by another convolutional layer.
			# Here, it reduces channel size from 128 to 64.
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
			nn.ReLU(),

			# Further upsampling and convolution.
			# This time, channel size is reduced from 64 to 32.
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
			nn.ReLU(),

			# Another upsampling and convolutional layer.
			# Reduces channel size from 32 to 16.
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
			nn.ReLU(),

			# Final convolutional layer that outputs n_channels.
			nn.Conv2d(16, self.n_channels, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),

			# Notes (original comments):
			# 16, 8, 16
			# 16, 16, 32
			# 8, 32, 64
			# 16, 64, 128
		)

		# Normalization occurs inside the model
		self.register_buffer('mu', torch.tensor([0., 0., 0., 0.]))
		self.register_buffer('std', torch.tensor([1., 1., 1., 1.]))


	def _num_parameters(self):
		count = 0
		for name, param in self.named_parameters():
			#print(name, param.numel())
			count += param.numel()
		return count

	def embed(self, x: Tensor) -> Tensor:
		"""Embeds tensor of state variables to LED observables

		Args:
			x (Tensor): [B, 2, H, W] Input feature tensor
			
		Returns:
			(Tensor): [B, config.n_embd] LED observables
		"""

		# x = self._normalize(x)
		# print("New x shape after normalization:", x.shape)

		g = self.observableNet(x)
		# print("g shape after observableNet:", g.shape)
		g = self.observableNetFC(g.view(x.size(0), -1))
		return g

	def recover(self, g: Tensor) -> Tensor:
		"""Recovers feature tensor from LED observables

		Args:
			g (Tensor): [B, config.n_embd] LED observables

		Returns:
			(Tensor): [B, 2, H, W] Physical feature tensor
		"""
		x = self.recoveryNet(g.view(-1, self.n_embd//256, 16, 16))
		# x = self._unnormalize(x)
		return x

	def _normalize(self, x: Tensor) -> Tensor:
		x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		return x

	def _unnormalize(self, x: Tensor) -> Tensor:
		return self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*x + self.mu[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
	
if __name__ == '__main__':
	n_embd = 1024
	layer_norm_epsilon = 1e-5
	embd_pdrop = 0.0
	n_channels = 2
	myModel = CNN_encoder_decoder_2D(n_embd, layer_norm_epsilon, embd_pdrop, n_channels)
	# Create mock inputs with specific dimensions
	B, H, W = 10, 512, 512  # Example batch size, height, and width
	x_mock = torch.zeros([B, n_channels, H, W])  # Replace with random or zeros

	print("Data dimensions before encoding:", x_mock.shape)

	# Call the embed function
	g = myModel.embed(x_mock)

	# Print the output dimensions
	print("Output dimensions after encoding:", g.shape)

	# Call the 'recover' function
	x_recovered = myModel.recover(g)

	# Print the output dimensions
	print("Output dimensions of recover:", x_recovered.shape)
	pdb.set_trace()