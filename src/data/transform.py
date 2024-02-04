import lightning as L
import torch
import torch.nn.functional as F


class Quantize(object):
	""" Custom Transform """

	def __init__(self, vqvae: L.LightningModule):
		self.vqvae = vqvae
		self.vqvae.eval()

	def __call__(self, x):
		return self.vqvae.get_quantized(x)[0]


class Masking(object):
	def __init__(self, probability: float, intra_source: bool = False, inter_source: bool = False):
		self.probability = probability
		self.intra_source = intra_source
		self.inter_source = inter_source

	def __call__(self, x: torch.Tensor):
		if self.intra_source:
			# intra-source masking: masking between different sound sources in a mixture
			rows_to_zero = torch.randperm(4)[:torch.randint(0, 3, (1,))]
			x[rows_to_zero, :] = 0

		if self.inter_source:
			# inter-source masking
			x = F.dropout(x, p=self.probability, training=True)

		return x
