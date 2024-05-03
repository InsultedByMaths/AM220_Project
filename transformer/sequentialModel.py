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

# This class is a Word specific fully connected neural network (FCNN) implementation,
# based on the architecture used in the GPT-2 model from Hugging Face's Transformers.
class MLP(nn.Module):
    '''
    Word specific FCNN implementation from:
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
    '''
    def __init__(self, n_state, config):  
        # n_state: The size of the hidden state (e.g., 3072 which is 4 times the embedding size in GPT-2)

        super().__init__()  # Initialize the superclass nn.Module

        nx = config.n_embd  # nx is the size of the embeddings

        # Define a 1D convolution layer to transform the input features to the hidden state size.
        # This layer acts as the first fully connected layer in the MLP.
        self.c_fc = Conv1D(n_state, nx)

        # Define another 1D convolution layer to transform the hidden state size back to the embedding size.
        # This layer acts as the second fully connected layer in the MLP.
        self.c_proj = Conv1D(nx, n_state)

        # Activation function: Retrieve the appropriate activation function from ACT2FN.
        self.act = ACT2FN[config.activation_function]

        # Dropout layer to prevent overfitting.
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        # Forward pass of the MLP.

        # Apply the first fully connected layer followed by the activation function.
        h = self.act(self.c_fc(x))

        # Apply the second fully connected layer.
        h2 = self.c_proj(h)

        # Apply dropout and return the result.
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
        the last dimension of the input will be operated on.
    """

    def __init__(self, nf, nx):
        # nf: Number of output features.
        # nx: Number of input features.

        super().__init__()  # Initialize the superclass nn.Module

        self.nf = nf  # Store the number of output features.

        # Initialize the weights: Create a tensor of shape (nx, nf) and apply normal initialization.
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)  # Register weight as a parameter of the model.

        # Initialize the biases: Create a bias vector of size nf, initialized to zero.
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # Forward pass of the Conv1D layer.

        # Calculate the output size. The last dimension will be replaced with the number of output features.
        size_out = x.size()[:-1] + (self.nf,)

        # Perform the linear operation: bias + (x * weight)
        # x is reshaped to a 2D tensor where the last dimension is the number of input features.
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)

        # Reshape x back to the original input size with the last dimension replaced by the number of output features.
        x = x.view(*size_out)

        return x

class Attention(nn.Module):
    """
    Implements the attention mechanism as described in the Transformer model.

    Args:
        nx (int): The number of embedding features (e.g., 128, 256, 512).
        n_ctx (int): The context length. Determines the size of the attention mask.
        config: A configuration object containing model settings.
        scale (bool): If True, scales the attention scores by 1/sqrt(dim_key).
    """
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        assert nx % config.n_head == 0  # Ensure the embedding size is divisible by the number of heads.
        
        # Create a lower triangular matrix to use as a bias for causal masking.
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        # Register a large negative value to use for masked positions.
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        
        self.n_head = config.n_head  # Number of attention heads.
        self.split_size = nx  # Size of each attention head.
        self.scale = scale  # Determines whether to scale the attention scores.

        # Convolutional layers for projecting input embeddings to query, key, and value vectors.
        self.c_attn = Conv1D(nx * 3, nx)
        self.c_proj = Conv1D(nx, nx)  # Convolutional layer for projecting the output of the attention mechanism.
        
        # Dropout layers for attention and residuals.
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        """
        Computes the attention scores and applies the attention mechanism.

        q, k, v: Query, key, and value tensors.
        attention_mask: Optional tensor to mask out certain positions from attending to others.
        head_mask: Optional tensor to mask out certain heads.
        output_attentions: If True, also returns the attention weights.
        """
        w = torch.matmul(q, k)  # Compute raw attention scores.
        if self.scale:
            # Scale attention scores to prevent large values due to dot product.
            w = w / (float(v.size(-1)) ** 0.5)
        
        # Apply causal mask to ensure each position can only attend to earlier positions.
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd: ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask provided.
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)  # Apply softmax to get attention probabilities.
        w = self.attn_dropout(w)  # Apply dropout to the attention scores.
        
        if head_mask is not None:
            w = w * head_mask  # Apply head mask if provided.

        outputs = [torch.matmul(w, v)]  # Compute the weighted sum of values.
        if output_attentions:
            outputs.append(w)  # Optionally return attention weights.
        
        return outputs  # Return the attention output and optionally the attention weights.

    def merge_heads(self, x):
        """
        Rearranges the attention output from (batch, head, seq_length, head_features)
        to (batch, seq_length, embedding_size).
        """
        x = x.permute(0, 2, 1, 3).contiguous()  # Rearrange axes.
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # Reshape to merge heads.

    def split_heads(self, x, k=False):
        """
        Splits the input embeddings into multiple heads for multi-head attention.

        k: If True, prepares the keys by transposing the last two dimensions.
        """
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # Reshape to split into heads.
        
        if k:
            return x.permute(0, 2, 3, 1)  # Transpose for keys.
        else:
            return x.permute(0, 2, 1, 3)  # Regular transpose for queries and values.

    def forward(self,
                x,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False):
        """
        Forward pass for the Attention layer.

        x: Input embeddings tensor.
        layer_past: Optional past key and value tensors for incremental decoding.
        attention_mask: Optional mask for blocking attention to certain positions.
        head_mask: Optional mask for disabling certain attention heads.
        use_cache: If True, returns key and value tensors for use in future incremental decoding steps.
        output_attentions: If True, also returns the attention weights.
        """
        x = self.c_attn(x)  # Apply the initial projection to get query, key, value.
        query, key, value = x.split(self.split_size, dim=2)  # Split the projection into q, k, v.
        
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)  # Split and transpose keys.
        # print(key)
        value = self.split_heads(value)
        
        if layer_past is not None:
            # Concatenate past keys and values for incremental decoding.
            past_key, past_value = layer_past
            # print(past_key)
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            present = torch.stack((key.transpose(-2, -1), value))  # Prepare tensors for caching.
        else:
            present = (None,)
        
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]  # Attention output.
        a = self.merge_heads(a)  # Merge attention heads.
        a = self.c_proj(a)  # Apply final projection.
        a = self.resid_dropout(a)  # Apply dropout to the output.
        
        outputs = [a, present] + attn_outputs[1:]  # Compile outputs.
        return outputs  # Return output tensors and optionally the attention weights and present state.

class Block(nn.Module):
    """
    Defines a block of the Transformer model, including an Attention layer followed by a feedforward neural network (MLP).

    Args:
        n_ctx (int): The context length for the attention mechanism.
        config: A configuration object containing model settings.
        scale (bool): If True, scales the attention scores by 1/sqrt(dim_key).
    """
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd  # Number of embedding features.
        
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)  # Pre-attention layer normalization.
        self.attn = Attention(nx, n_ctx, config, scale)  # Attention layer.
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)  # Post-attention layer normalization.
        self.mlp = MLP(4 * nx, config)  # Feedforward neural network.

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        """
        Forward pass for the Transformer block.

        x: Input embeddings tensor.
        layer_past: Optional past key and value tensors for incremental decoding.
        attention_mask: Optional mask for blocking attention to certain positions.
        head_mask: Optional mask for disabling certain attention heads.
        use_cache: If True, enables incremental decoding by caching key and value tensors.
        output_attentions: If True, also returns the attention weights.
        """
        attn_output = self.attn(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        a = attn_output[0]  # Attention output.
        x = x + a  # Apply residual connection.
        
        mlp_output = self.mlp(self.ln_2(x))  # Pass through the MLP.
        x = x + mlp_output  # Apply second residual connection.
        
        outputs = [x] + attn_output[1:]  # Compile outputs.
        return outputs  # Return the final output tensor, along with any optional states and attention weights.

class SequentialModel(nn.Module):
    """
    A sequential model that incorporates multiple Transformer blocks.

    Args:
        config: A configuration object containing all necessary parameters for the model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # Model configuration
        self.output_hidden_states = config.output_hidden_states  # Whether to output hidden states
        self.drop = nn.Dropout(config.embd_pdrop)  # Dropout layer for embeddings

        # A sequence of Transformer blocks as defined by the Block class
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)  # Layer normalization at the output
        self.mlp_f = nn.Linear(config.n_embd, config.n_embd)  # Final linear layer
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)  # Positional embeddings
        self.init_weights()  # Initialize weights
        self.n_embd = config.n_embd  # Embedding dimension size

    def init_weights(self):
        """
        Initializes weights of the model with normal distribution and sets biases to zero.
        """
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
        """
        Counts the number of parameters in the model.
        """
        count = 0
        for _, param in self.named_parameters():
            count += param.numel()
        return count
    
    def forward(self,
                inputs_embeds,
                past=None,
                attention_mask=None,
                position_ids=None,
                prop_embeds=None,
                head_mask=None,
                use_cache=True,
                output_attentions=None):
        """
        Forward pass of the model.

        Args:
            inputs_embeds: Input embeddings.
            past: Optional past key values for each layer.
            attention_mask: Optional attention mask.
            position_ids: Optional position ids.
            prop_embeds: Unused, for future use.
            head_mask: Optional mask for attention heads.
            use_cache: If True, uses caching for faster decoding.
            output_attentions: If True, outputs attentions of each layer.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        # Position IDs handling
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.float, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]).repeat(batch_size, 1)

        # Attention mask processing
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
			# Sizes are [batch_size, 1, 1, to_seq_length]
			# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
			# this attention mask is more simple than the triangular masking of causal attention
			# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.view(batch_size, -1).unsqueeze(1).unsqueeze(2)
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
			# masked positions, this operation will create a tensor which is 0.0 for
			# positions we want to attend and -10000.0 for masked positions.
			# Since we are adding it to the raw scores before the softmax, this is
			# effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Positional embeddings generation
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        position_embeds = self.generate_position_embeddings(inputs_embeds, position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()  # To store outputs for caching
        all_attentions = []
        all_hidden_states = ()

        # Process each block
        for i, (block, layer_past) in enumerate(zip(self.h, past or [None] * len(self.h))):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        # Apply final layer norm and linear layer
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.mlp_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        # Prepare outputs
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        return outputs  # Outputs may include last hidden state, cached keys/values, all hidden states, and attentions

    def generate_position_embeddings(self, inputs_embeds, position_ids):
        """
        Generates sinusoidal position embeddings.
        """
        position_embeds = torch.zeros_like(inputs_embeds)
        i = torch.arange(0, self.config.n_embd // 2, dtype=torch.float, device=inputs_embeds.device).unsqueeze(0).unsqueeze(0)
        div_term = 10000 ** (2 * i / self.config.n_embd)
        position_embeds[..., ::2] = torch.sin(position_ids[..., None] / div_term)
        position_embeds[..., 1::2] = torch.cos(position_ids[..., None] / div_term)
        return position_embeds

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
    def __init__(self, config):
        """
        Initializes the composite model consisting of a SequentialModel and an MLPDense model.

        Args:
            config: A configuration object containing model settings and parameters.
        """
        super().__init__()  # Initialize the superclass nn.Module
        
        # Initialize the SequentialModel component of the composite model
        # This part of the model handles sequence-level processing.
        self.seqModel = SequentialModel(config)
        
        # Initialize the MLP (Multilayer Perceptron) component of the composite model
        # here it is tasked with enriching token representations or processing individual tokens.
        # The input dimension is config.paraEnrichDim, and the output dimension is config.n_embd.
        # The MLP has two hidden layers, each with 200 units, and includes ReLU activations.
        self.TokenModel = MLPDense(config.paraEnrichDim, config.n_embd, [200, 200], True)

    def _num_parameters(self):
        """
        Calculates the total number of parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        count = 0
        # Iterate over all parameters in the model, summing their total number
        for _, param in self.named_parameters():
            count += param.numel()
        return count


if __name__ == '__main__':
	print('I love you.')
	#x = torch.randn(batch_size, n_steps, config.n_embd) # Batch, time-steps, embed