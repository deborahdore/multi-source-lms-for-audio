from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import torchaudio
import wandb
from torch import nn, optim


class VQVAE(L.LightningModule):
	def __init__(self,
				 num_hidden: int,
				 num_residual_layer: int,
				 num_residual_hidden: int,
				 num_embedding: int,
				 embedding_dim: int,
				 commitment_cost: float,
				 learning_rate: int,
				 output_dir: str):
		super(VQVAE, self).__init__()

		self.encoder = Encoder(in_channel=1,
							   num_hidden=num_hidden,
							   num_residual_layer=num_residual_layer,
							   num_residual_hidden=num_residual_hidden)

		self.conv = nn.Conv1d(in_channels=num_hidden, out_channels=embedding_dim, kernel_size=1, stride=1)

		self.vector_quantizer = VectorQuantizer(num_embedding=num_embedding,
												embedding_dim=embedding_dim,
												commitment_cost=commitment_cost)

		self.decoder = Decoder(in_channel=embedding_dim,
							   num_hidden=num_hidden,
							   num_residual_layer=num_residual_layer,
							   num_residual_hidden=num_residual_hidden)

		self.learning_rate = learning_rate
		self.output_dir = output_dir
		self.save_hyperparameters()

	def training_step(self, batch, batch_idx):
		mixed, instruments = batch
		output, loss = self.forward(mixed)

		# loss per instruments
		for i in range(instruments.size(-2)):
			loss += F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])

		# loss on full audio
		mixed_output = torch.einsum('bij-> bj', output)
		loss += F.mse_loss(input=mixed_output, target=mixed.squeeze(1))

		self.log("train_loss", loss, on_epoch=True, sync_dist=True)

		return loss

	def forward(self, x):
		z = self.conv(self.encoder(x))
		loss, quantized, perplexity, _ = self.vector_quantizer(z)
		output = self.decoder(quantized)
		return output, loss

	def validation_step(self, batch, batch_idx):
		mixed, instruments = batch
		output, loss = self.forward(mixed)

		# loss per instruments
		for i in range(instruments.size(-2)):
			loss += F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])

		# loss on full audio
		mixed_output = torch.einsum('bij-> bj', output)
		loss += F.mse_loss(input=mixed_output, target=mixed.squeeze(1))

		self.log("val_loss", loss, on_epoch=True, sync_dist=True)

		return loss

	def test_step(self, batch, batch_idx):
		mixed, instruments = batch
		output, loss = self.forward(mixed)

		# loss per instruments
		for i in range(instruments.size(-2)):
			loss += F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])

		# loss on full audio
		mixed_output = torch.einsum('bij-> bj', output)
		loss += F.mse_loss(input=mixed_output, target=mixed.squeeze(1))

		self.log("test_loss", loss, on_epoch=True, sync_dist=True)

		return loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, amsgrad=False)
		return optimizer

	def on_validation_batch_end(self, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		# only log on the first batch of validation
		if batch_idx != 0: return

		instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]

		with torch.no_grad():
			mixed, instruments = batch
			mixed = mixed[0]
			instruments = instruments[0]
			output_instruments, _ = self.forward(mixed.unsqueeze(0))
			output_instruments = output_instruments.squeeze()

			epoch = self.trainer.current_epoch
			sample_rate = self.trainer.val_dataloaders.dataset.target_sample_rate

			data = [[], []]
			for idx in range(instruments.size(0)):
				original_file = f'{self.output_dir}/original_{epoch}_{instruments_name[idx]}'
				decoded_file = f'{self.output_dir}/generated_{epoch}_{instruments_name[idx]}'

				torchaudio.save(uri=original_file,
								src=instruments[idx].unsqueeze(0).detach().cpu(),
								sample_rate=sample_rate)

				torchaudio.save(uri=decoded_file,
								src=output_instruments[idx].unsqueeze(0).detach().cpu(),
								sample_rate=sample_rate)

				data[0].append(wandb.Audio(str(original_file), sample_rate=sample_rate))
				data[1].append(wandb.Audio(str(decoded_file), sample_rate=sample_rate))

			original_full_file = f'{self.output_dir}/original_{epoch}_full_song.wav'
			decoded_full_file = f'{self.output_dir}/generated_{epoch}_full_song.wav'

			torchaudio.save(uri=original_full_file, src=mixed.detach().cpu(), sample_rate=sample_rate)
			torchaudio.save(uri=decoded_full_file,
							src=torch.einsum('ij-> j', output_instruments).unsqueeze(0).detach().cpu(),
							sample_rate=sample_rate)

			data[0].append(wandb.Audio(str(original_full_file), sample_rate=sample_rate))
			data[1].append(wandb.Audio(str(decoded_full_file), sample_rate=sample_rate))

			if isinstance(self.logger, L.pytorch.loggers.wandb.WandbLogger):
				columns = ['bass vs D(bass)', 'drums vs D(drums)', 'guitar vs D(guitar)', 'piano vs D(piano)',
						   'mixed vs D(mixed)']
				table = wandb.Table(columns=columns, data=data)
				self.logger.log_table({f'demos_{epoch}': table})


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


class VectorQuantizer(nn.Module):
	"""
	https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
	"""

	def __init__(self, num_embedding: int, embedding_dim: int, commitment_cost: float):
		super(VectorQuantizer, self).__init__()

		self.embedding_dim = embedding_dim
		self.num_embedding = num_embedding

		self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
		self.embedding.weight.data.uniform_(-1 / self.num_embedding, 1 / self.num_embedding)
		self.commitment_cost = commitment_cost

	def forward(self, inputs):
		# convert from BCW -> BWC
		inputs = torch.einsum('bcw -> bwc', inputs).contiguous()
		input_shape = inputs.shape

		# Flatten input
		flat_input = inputs.view(-1, self.embedding_dim)

		# Compute L2 distance between latents and embedding weights
		distances = torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2,
																				dim=1) - 2 * torch.matmul(flat_input,
																										  self.embedding.weight.t())

		# Encoding
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embedding, device=inputs.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantize and unflatten
		quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

		# Loss
		commitment_loss = F.mse_loss(quantized.detach(), inputs)
		embedding_loss = F.mse_loss(quantized, inputs.detach())
		loss = embedding_loss + self.commitment_cost * commitment_loss

		quantized = inputs + (quantized - inputs).detach()
		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		# convert from BWC -> BCW
		return loss, torch.einsum('bwc -> bcw', quantized).contiguous(), perplexity, encodings


class Encoder(nn.Module):
	def __init__(self, in_channel: int, num_hidden: int, num_residual_layer: int, num_residual_hidden: int):
		super(Encoder, self).__init__()

		self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=num_hidden // 2, kernel_size=4, stride=2,
							   padding=1)

		self.conv2 = nn.Conv1d(in_channels=num_hidden // 2, out_channels=num_hidden, kernel_size=4, stride=2,
							   padding=1)

		self.conv3 = nn.Conv1d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

		self.residual_stack = ResidualStack(in_channel=num_hidden,
											num_hidden=num_hidden,
											num_residual_layer=num_residual_layer,
											num_residual_hidden=num_residual_hidden)

	def forward(self, x):
		# initial dimension: BCW
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.conv3(x)
		return self.residual_stack(x)


class Decoder(nn.Module):
	def __init__(self, in_channel: int, num_hidden: int, num_residual_layer: int, num_residual_hidden: int):
		super(Decoder, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

		self.residual_stack = ResidualStack(in_channel=num_hidden,
											num_hidden=num_hidden,
											num_residual_layer=num_residual_layer,
											num_residual_hidden=num_residual_hidden)

		self.conv1_transpose = nn.ConvTranspose1d(in_channels=num_hidden,
												  out_channels=num_hidden // 2,
												  kernel_size=4,
												  stride=2,
												  padding=1)

		self.conv2_transpose = nn.ConvTranspose1d(in_channels=num_hidden // 2,
												  out_channels=4,
												  kernel_size=4,
												  stride=2,
												  padding=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.residual_stack(x)
		x = F.relu(self.conv1_transpose(x))
		return self.conv2_transpose(x)
