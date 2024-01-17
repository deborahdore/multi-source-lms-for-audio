from scr.model.vqvae import VQVAE


class Quantize(object):
	""" Custom Transform """

	def __init__(self, vqvae: VQVAE):
		self.vqvae = vqvae
		self.vqvae.eval()

	def __call__(self, x):
		return self.vqvae.get_quantized(x)[0]
