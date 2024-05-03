import torch
import torch.nn as nn
import pdb
from dataclasses import dataclass
import torch.nn.functional as F
import time
import numpy as np
import math
import torch.nn.functional as F
from packaging import version
from spatialModel import MLP as MLPDense

class MLP(nn.Module):
	'''
	Word specific FCNN implementation from:
	https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
	'''
	def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
		super().__init__()
		nx = config.n_embd
		self.c_fc = Conv1D(n_state, nx)
		self.c_proj = Conv1D(nx, n_state)
		self.act = ACT2FN[config.activation_function]
		self.dropout = nn.Dropout(config.resid_pdrop)

	def forward(self, x):
		h = self.act(self.c_fc(x))
		h2 = self.c_proj(h)
		return self.dropout(h2)

class Conv1D(nn.Module):
	"""
	1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
	Basically works like a linear layer but the weights are transposed.
	Args:
		nf (:obj:`int`): The number of output features.
		nx (:obj:`int`): The number of input features.
	Note:
		When the model is used for forward propagation,
		the last dimension of the input will be operate  
	"""

	def __init__(self, nf, nx):
		super().__init__()
		self.nf = nf
		w = torch.empty(nx, nf)
		nn.init.normal_(w, std=0.02)
		self.weight = nn.Parameter(w)
		self.bias = nn.Parameter(torch.zeros(nf))

	def forward(self, x):
		size_out = x.size()[:-1] + (self.nf,)
		x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
		x = x.view(*size_out)
		return x

class Attention(nn.Module):
	"""
	Args:
		nx (:obj:`int`): The number of embedding feature, e.g., 128, 256, 512 or so
		n_ctx (:obj:`int`): The context length (not sure)
		config (:obj:T.B.D):
	"""
	def __init__(self, nx, n_ctx, config, scale=False):
		super().__init__()
		
		assert nx % config.n_head == 0
		self.register_buffer(
			"bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
		)
		self.register_buffer("masked_bias", torch.tensor(-1e4))
		self.n_head = config.n_head
		self.split_size = nx
		self.scale = scale

		self.c_attn = Conv1D(nx * 3, nx) # Kindly reminder: input_size = [..., nx] and output_size = [..., 3 * nx]
		self.c_proj = Conv1D(nx, nx) # Question: what is the use of this self.c_proj?
		self.attn_dropout = nn.Dropout(config.attn_pdrop)
		self.resid_dropout = nn.Dropout(config.resid_pdrop)

	def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
		w = torch.matmul(q, k)
		if self.scale:
			w = w / (float(v.size(-1)) ** 0.5)
		nd, ns = w.size(-2), w.size(-1)
		mask = self.bias[:, :, ns - nd : ns, :ns]
		w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

		if attention_mask is not None:
			# Apply the attention mask
			w = w + attention_mask

		w = nn.Softmax(dim=-1)(w)
		w = self.attn_dropout(w)
		
		# Mask heads if we want to
		if head_mask is not None:
			w = w * head_mask

		outputs = [torch.matmul(w, v)]
		if output_attentions:
			outputs.append(w)
		return outputs # [value, weights]
	
	def merge_heads(self, x):
		x = x.permute(0, 2, 1, 3).contiguous()
		new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
		return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

	def split_heads(self, x, k=False):
		new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
		
		x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
		if k:
			return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
		else:
			return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
		

	def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
		x = self.c_attn(x) # x -> q, k, v
		query, key, value = x.split(self.split_size, dim=2) # HanGao: wouldn't it be more general if dim=-1?
		
		query = self.split_heads(query)
		
		key = self.split_heads(key, k=True) # k=True for keys which transposes the last two dims
		value = self.split_heads(value)
		# Concat previous key and value tensors 
		if layer_past is not None:
			past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
			key = torch.cat((past_key, key), dim=-1)
			value = torch.cat((past_value, value), dim=-2)

		if use_cache is True:
			present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
		else:
			present = (None,)
		
		attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
		
		a = attn_outputs[0]
		a = self.merge_heads(a)
		
		a = self.c_proj(a)
		a = self.resid_dropout(a)
		
		outputs = [a, present] + attn_outputs[1:]
		return outputs

class Block(nn.Module):
	def __init__(self, n_ctx, config, scale=False):
		super().__init__()
		nx = config.n_embd
		self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.attn = Attention(nx, n_ctx, config, scale)
		self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.mlp = MLP(4 * nx, config)

	def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
		# Evaluate attention heads
		output_attn = self.attn.forward(
			self.ln_1(x),
			layer_past=layer_past,
			attention_mask=attention_mask,
			head_mask=head_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
		)
		a = output_attn[0]  # output_attn: a, present, (attentions)
		# Residual connection 1
		x = x + a
		# FCNN
		m = self.mlp(self.ln_2(x))
		# Residual connection 2
		x = x + m

		outputs = [x] + output_attn[1:]
		return outputs  # x, present, (attentions)

class SequentialModel(nn.Module):
	def __init__(self, config):  # in MLP: n_state=3072 (4 * n_embd)
		super().__init__()
		self.config = config
		self.output_hidden_states = config.output_hidden_states
		self.drop = nn.Dropout(config.embd_pdrop)
		self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
		self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
		self.mlp_f = nn.Linear(config.n_embd, config.n_embd)
		self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
		self.init_weights()
		self.n_embd = config.n_embd
	
	def init_weights(self):
		for module in self.modules():
			if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
				if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.LayerNorm):
				module.bias.data.zero_()
				module.weight.data.fill_(1.0)
	def _num_parameters(self):
		count = 0
		for name, param in self.named_parameters():
			#print(name, param.numel())
			count += param.numel()
		return count
	
	def forward(self,
			inputs_embeds=None,      # Embeddings of input tokens. 
			past=None,               # Contains past key and value states.
			attention_mask=None,     # A mask to avoid paying attention to certain tokens. 
			position_ids=None,       # Specifies the position indices for each input token.
			prop_embeds=None,        
			head_mask=None,          # A mask to nullify selected heads of the multi-head attention modules.
			use_cache=True,          # Indicates whether to use caching of past key and value states. 
			output_attentions=None   # Whether to return attention weights. 
			):
		
		# Decide whether to output attention weights based on the provided argument or model's configuration.
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

		# Determine the shape of the input embeddings, excluding the embedding dimension.
		input_shape = inputs_embeds.size()[:-1]

		# Extract the batch size from the input embeddings' shape.
		batch_size = inputs_embeds.shape[0]

		# Check if position_ids were provided explicitly.
		if position_ids is not None:
			# Reshape the position_ids to match the input shape, ensuring compatibility.
			position_ids = position_ids.view(-1, input_shape[-1])

		
		"""
		Han Gao: prop_embeds has no use!
		"""

		# Check if property embeddings are provided.
		if prop_embeds is not None:
			# Ensure the batch size of property embeddings matches that of the input embeddings.
			assert inputs_embeds.size(0) == prop_embeds.size(0), 'Property embeddings do not match the size of the input'
			# Adjust the property embeddings to match the sequence length of the input embeddings.
			prop_embeds = prop_embeds[:, :inputs_embeds.size(1)]
		else:
			# If no property embeddings are provided, initialize them as zeros with the same shape as the input embeddings.
			prop_embeds = torch.zeros_like(inputs_embeds)

		# Check if a 'past' state is provided.
		if past is None:
			# If there is no 'past' state, set the past_length to 0 indicating no previously computed states.
			past_length = 0
			# Initialize 'past' as a list of None values, one for each layer in the transformer model.
			# 'self.h' represents the layers of the transformer, so its length determines the number of past states to initialize.
			past = [None] * len(self.h)
		else:
			# If 'past' is provided, calculate the length of the past sequence.
			# 'past[0][0]' accesses the first layer's past state, and '.size(-2)' gets the sequence length of this state.
			past_length = past[0][0].size(-2)
			
		# Check if position_ids were not provided.
		if position_ids is None:
			# Get the device from the input embeddings to ensure compatibility (e.g., CPU or GPU).
			device = inputs_embeds.device
			
			# Generate a range of position IDs starting from past_length to the end of the current input sequence.
			position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.float, device=device)
			
			# Reshape the position_ids to have a batch dimension of 1, then adjust to match the input sequence length.
			position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
			
			# Repeat the position_ids for each example in the batch to match the batch size of inputs_embeds.
			position_ids = position_ids.repeat(inputs_embeds.size(0), 1)

			
		# Attention mask.
		if attention_mask is not None:
			assert batch_size > 0, "batch_size has to be defined and > 0"
			attention_mask = attention_mask.view(batch_size, -1)
			# We create a 3D attention mask from a 2D tensor mask.
			# Sizes are [batch_size, 1, 1, to_seq_length]
			# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
			# this attention mask is simpler than the triangular masking of causal attention
			# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
			attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

			# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
			# masked positions, this operation will create a tensor which is 0.0 for
			# positions we want to attend and -10000.0 for masked positions.
			# Since we are adding it to the raw scores before the softmax, this is
			# effectively the same as removing these entirely.
			attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
			attention_mask = (1.0 - attention_mask) * -10000.0
			

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# head_mask has shape n_layer x batch x n_heads x N x N
		# head_mask = self.get_head_mask(head_mask, self.config.n_layer)

		# If embeddings are not given as the input, embed the provided word ids
		# position_embeds = self.wpe(position_ids)

		# Function embeddings
		# http://papers.nips.cc/paper/7181-attention-is-all-you-need
		# Initialize position embeddings tensor with zeros, matching the shape of input embeddings.
		position_embeds = torch.zeros_like(inputs_embeds)

		# Generate indices for the dimension of position embeddings, halved because of sine and cosine alternation.
		# The unsqueeze operations add extra dimensions to fit the shape required for broadcasting in subsequent operations.
		i = torch.arange(0, self.config.n_embd // 2, dtype=torch.float, device=inputs_embeds.device).unsqueeze(0).unsqueeze(0)

		# Apply sine function to even indices of position embeddings.
		# The division term scales the frequency of the sine wave, with different frequencies for each dimension.
		position_embeds[:, :, ::2] = torch.sin(position_ids.unsqueeze(-1) / 10000 ** (2 * i.type(torch.FloatTensor).to(inputs_embeds.device) / self.config.n_embd))

		# Adjust 'i' for cosine application by considering the embedding dimension's parity.
		i = i[:, :, self.config.n_embd % 2]

		# Apply cosine function to odd indices of position embeddings.
		position_embeds[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1) / 10000 ** (2 * i.type(torch.FloatTensor).to(inputs_embeds.device) / self.config.n_embd))

		# Combine the input embeddings with the generated position embeddings.
		hidden_states = inputs_embeds + position_embeds

		# Apply dropout to the combined embeddings to prevent overfitting.
		hidden_states = self.drop(hidden_states)


		# Construct the output shape by extending the input shape with the size of the last dimension of hidden_states.
		output_shape = input_shape + (hidden_states.size(-1),)


		# Initialize empty tuples for caching the outputs and empty list for attention weights.
		presents = ()
		all_attentions = []
		all_hidden_states = ()

		# Iterate over each transformer block and the corresponding past state.
		for i, (block, layer_past) in enumerate(zip(self.h, past)):
			# If configured to output hidden states, append the current hidden states to the collection.
			if self.output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

			# Call the current transformer block.
			outputs = block(
				hidden_states,
				layer_past=layer_past,
				attention_mask=attention_mask,
				use_cache=use_cache,
				output_attentions=output_attentions,
			)

			# Extract the main output (new hidden states) and the present state from the block's output.
			hidden_states, present = outputs[:2]
			
			# If using cache, append the present state to the 'presents' tuple for future use.
			if use_cache is True:
				presents = presents + (present,)

			# If outputting attentions, append the attention weights to the 'all_attentions' list.
			if output_attentions:
				all_attentions.append(outputs[2])


		# Apply layer normalization to the hidden states.
		hidden_states = self.ln_f(hidden_states)

		# Pass the normalized hidden states through the MLP (multi-layer perceptron) feedforward network.
		hidden_states = self.mlp_f(hidden_states)

		# hidden_states = self.mlp_f(self.ln_f(hidden_states).view(-1, self.n_embd // 64, 64))

		# Reshape the hidden states to match the original input shape with the added dimension for features.
		hidden_states = hidden_states.view(*output_shape)

		# Conditionally add the last hidden state to the collection of all hidden states.
		if self.output_hidden_states:
			# If the model is configured to output hidden states, append the final hidden states to the collection.
			all_hidden_states = all_hidden_states + (hidden_states,)


		# Initialize the output tuple with the main output: hidden states.
		outputs = (hidden_states,)

		# If caching is enabled, append the cached states to the output.
		if use_cache is True:
			outputs = outputs + (presents,)

		# If the model is configured to output hidden states, append them to the output.
		if self.output_hidden_states:
			outputs = outputs + (all_hidden_states,)

		# If the model is configured to output attention weights, process and append them.
		if output_attentions:
			# Define the shape for the attention outputs, allowing for dynamic number of heads.
			attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
			
			# Reshape all attention matrices to match the defined shape.
			all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
			
			# Append the processed attention weights to the output.
			outputs = outputs + (all_attentions,)

		# Han Gao: We must figure out what dose the output contain?	
		return outputs # [last_hidden_state, past_key_values, hidden_states, attentions]



def _gelu_python(x):
	"""
	Original Implementation of the GELU activation function in Google BERT repo when initially created. For
	information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
	torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in
	torch.nn.functional Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
	"""
	Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
	the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
	"""
	return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if version.parse(torch.__version__) < version.parse("1.4"):
	gelu = _gelu_python
else:
	gelu = F.gelu


def gelu_fast(x):
	return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def _silu_python(x):
	"""
	See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
	Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
	Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
	Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
	later.
	"""
	return x * torch.sigmoid(x)


if version.parse(torch.__version__) < version.parse("1.7"):
	silu = _silu_python
else:
	silu = F.silu


def mish(x):
	return x * torch.tanh(torch.nn.functional.softplus(x))


def linear_act(x):
	return x


ACT2FN = {
	"relu": F.relu,
	"silu": silu,
	"swish": silu,
	"gelu": gelu,
	"tanh": torch.tanh,
	"gelu_new": gelu_new,
	"gelu_fast": gelu_fast,
	"mish": mish,
	"linear": linear_act,
	"sigmoid": torch.sigmoid,
}


def get_activation(activation_string):
	if activation_string in ACT2FN:
		return ACT2FN[activation_string]
	else:
		raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))


class SeqModelWithMLP(nn.Module):
	def __init__(self, config):  # in MLP: n_state=3072 (4 * n_embd)
		super().__init__()
		self.seqModel = SequentialModel(config)
		self.TokenModel = MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
	def _num_parameters(self):
		count = 0
		for name, param in self.named_parameters():
			#print(name, param.numel())
			count += param.numel()
		return count


if __name__ == '__main__':
	print('I love you.')
	#x = torch.randn(batch_size, n_steps, config.n_embd) # Batch, time-steps, embed