import torch
from torch import nn as nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
	"""
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	"""

	def __init__(self, num_embedding: int, embedding_dim: int, commitment_cost: float):
		super(VectorQuantizer, self).__init__()

		self.embedding_dim = embedding_dim
		self.num_embedding = num_embedding

		# codebook
		self.codebook = nn.Embedding(self.num_embedding, self.embedding_dim)
		self.codebook.weight.data.uniform_(-1 / self.num_embedding, 1 / self.num_embedding)

		self.commitment_cost = commitment_cost

	def forward(self, inputs):
		# convert from BCW -> BWC
		inputs = torch.einsum('bcw -> bwc', inputs).contiguous()
		input_shape = inputs.shape

		# Flatten input
		flat_input = inputs.view(-1, self.embedding_dim)

		# Compute L2 distance between latents and embedding weights
		distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)) + (
				torch.sum(self.codebook.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self.codebook.weight.t()))

		# Encoding
		# For every element in the input, find the closest embedding
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embedding, device=inputs.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantize and unflatten
		quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

		# Loss
		commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
		embedding_loss = F.mse_loss(quantized, inputs.detach())

		quantized = inputs + (quantized - inputs).detach()
		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		# convert from BWC -> BCW
		quantized = torch.einsum('bwc -> bcw', quantized).contiguous()

		return embedding_loss, commitment_loss, quantized, perplexity, encodings, encoding_indices


class ResidualStack(nn.Module):
	def __init__(self, in_channel: int, num_hidden: int, num_residual_layer: int, num_residual_hidden: int):
		super(ResidualStack, self).__init__()

		self.residual_layers = nn.ModuleList([nn.Sequential(nn.ReLU(True),
															nn.Conv1d(in_channels=in_channel if i == 0 else num_hidden,
																	  out_channels=num_residual_hidden,
																	  kernel_size=3,
																	  stride=1,
																	  padding=1,
																	  bias=False),
															nn.ReLU(True),
															nn.Conv1d(in_channels=num_residual_hidden,
																	  out_channels=num_hidden,
																	  kernel_size=1,
																	  stride=1,
																	  bias=False)) for i in range(num_residual_layer)])

	def forward(self, x):
		for layer in self.residual_layers:
			x = x + layer(x)
		return F.relu(x)
