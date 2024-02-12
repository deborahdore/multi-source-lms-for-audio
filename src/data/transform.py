import lightning as L
import torch


class Quantize(object):
	""" Custom Transform """

	def __init__(self, vqvae: L.LightningModule):
		self.vqvae = vqvae
		self.vqvae.eval()

	def get_quantized(self, x):
		return self.vqvae.get_quantized(x)[0]

	def get_encodings_idx(self, x):
		return self.vqvae.get_quantized(x)[2]


class ToComplex(torch.nn.Module):
	""" Custom Transform """

	def __call__(self, x: torch.Tensor):
		return x.to(torch.cfloat)
