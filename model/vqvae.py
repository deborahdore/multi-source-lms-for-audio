from typing import Any

import lightning as L
import torch
import torchaudio
import wandb
from torch import nn as nn, optim
from torch.nn import functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from model.decoder import Decoder
from model.encoder import Encoder
from model.modules import VectorQuantizer


class VQVAE(L.LightningModule):
	def __init__(self,
				 num_hidden: int,
				 num_residual_layer: int,
				 num_residual_hidden: int,
				 num_embedding: int,
				 embedding_dim: int,
				 commitment_cost: float,
				 learning_rate: int,
				 checkpoint_dir: str):
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
		self.checkpoint_dir = checkpoint_dir
		self.best_loss = float('inf')
		self.save_hyperparameters()

	def training_step(self, batch, batch_idx):
		mixed, instruments = batch

		# commitment loss + embedding loss
		output, embedding_loss, commitment_loss = self.forward(mixed)

		loss = embedding_loss + commitment_loss

		# loss per instrument
		for i in range(4):
			# TRAINING WITH L1 LOSS
			loss += F.l1_loss(input=output[:, i, :], target=instruments[:, i, :])

		self.log("train/loss", loss, on_epoch=True)
		return loss

	def forward(self, x):
		z = self.conv(self.encoder(x))
		embedding_loss, commitment_loss, quantized, perplexity, encodings = self.vector_quantizer(z)
		output = self.decoder(quantized)
		return output, embedding_loss, commitment_loss

	@torch.no_grad()
	def get_quantized(self, x):
		z = self.conv(self.encoder(x))
		_, quantized, _, _ = self.vector_quantizer(z)
		return quantized

	def validation_step(self, batch, batch_idx):
		mixed, instruments = batch

		output, embedding_loss, commitment_loss = self.forward(mixed)
		mixed_output = torch.einsum('bij-> bj', output)

		instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]

		# commitment loss
		self.log("validation/embedding_loss", embedding_loss, on_epoch=True)

		# embedding loss
		self.log("validation/commitment_loss", commitment_loss, on_epoch=True)

		loss = embedding_loss + commitment_loss

		# loss per instrument
		l1_instrument_loss = 0
		for i, instrument in enumerate(instruments_name):
			l1_instruments_loss = F.l1_loss(input=output[:, i, :], target=instruments[:, i, :])
			self.log(f"validation/l1_{instrument}_loss", l1_instruments_loss, on_epoch=True)

			self.log(f"validation/l2_{instrument}_loss",
					 F.mse_loss(input=output[:, i, :], target=instruments[:, i, :]),
					 on_epoch=True)

			self.log(f"validation/si_sdr_{instrument}_loss",
					 scale_invariant_signal_distortion_ratio(preds=output[:, i, :], target=instruments[:, i,
																						   :]).mean(),
					 on_epoch=True)

			l1_instrument_loss += l1_instruments_loss

		self.log("validation/l1_instruments_loss", l1_instrument_loss, on_epoch=True)

		# SI-SDR loss on combined audio
		self.log("validation/si_sdr_full_audio_loss",
				 scale_invariant_signal_distortion_ratio(preds=mixed_output, target=mixed.squeeze(1)).mean(),
				 on_epoch=True)

		# MSE loss combined audio
		self.log("validation/l2_full_audio_loss",
				 F.mse_loss(input=mixed_output, target=mixed.squeeze(1)),
				 on_epoch=True)

		# L1 loss combined audio
		self.log("validation/l1_full_audio_loss", F.l1_loss(input=mixed_output, target=mixed.squeeze(1)),
				 on_epoch=True)

		loss += l1_instrument_loss
		self.log("validation/loss", loss, on_epoch=True)
		return loss

	def test_step(self, batch, batch_idx):
		mixed, instruments = batch

		output, embedding_loss, commitment_loss = self.forward(mixed)
		mixed_output = torch.einsum('bij-> bj', output)

		instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]

		# commitment loss
		self.log("test/embedding_loss", embedding_loss, on_epoch=True)

		# embedding loss
		self.log("test/commitment_loss", commitment_loss, on_epoch=True)

		loss = embedding_loss + commitment_loss

		# loss per instrument
		l1_instrument_loss = 0
		for i, instrument in enumerate(instruments_name):
			l1_instruments_loss = F.l1_loss(input=output[:, i, :], target=instruments[:, i, :])
			self.log(f"test/l1_{instrument}_loss", l1_instruments_loss, on_epoch=True)

			self.log(f"test/l2_{instrument}_loss",
					 F.mse_loss(input=output[:, i, :], target=instruments[:, i, :]),
					 on_epoch=True)

			self.log(f"test/si_sdr_{instrument}_loss",
					 scale_invariant_signal_distortion_ratio(preds=output[:, i, :], target=instruments[:, i,
																						   :]).mean(),
					 on_epoch=True)

			l1_instrument_loss += l1_instruments_loss

		self.log("test/l1_instruments_loss", l1_instrument_loss, on_epoch=True)

		# SI-SDR loss on combined audio
		self.log("test/si_sdr_full_audio_loss",
				 scale_invariant_signal_distortion_ratio(preds=mixed_output, target=mixed.squeeze(1)).mean(),
				 on_epoch=True)

		# MSE loss combined audio
		self.log("test/l2_full_audio_loss", F.mse_loss(input=mixed_output, target=mixed.squeeze(1)), on_epoch=True)

		# L1 loss combined audio
		self.log("test/l1_full_audio_loss", F.l1_loss(input=mixed_output, target=mixed.squeeze(1)), on_epoch=True)

		loss += l1_instrument_loss
		self.log("test/loss", loss, on_epoch=True)
		return loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, amsgrad=False)
		return optimizer

	def on_validation_batch_end(self, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		try:
			# only log on the first batch of validation
			if batch_idx != 0:
				return

			if outputs.item() < self.best_loss:
				self.best_loss = outputs.item()
				self.log("validation/best_loss", self.best_loss, on_epoch=True)

			if not isinstance(self.logger, L.pytorch.loggers.wandb.WandbLogger):
				return

			instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]

			with torch.no_grad():
				mixed, instruments = batch
				mixed = mixed[0]
				instruments = instruments[0]
				output_instruments, _ = self.forward(mixed.unsqueeze(0))
				output_instruments = output_instruments.squeeze()

				sample_rate = self.trainer.val_dataloaders.dataset.target_sample_rate
				epoch = self.trainer.current_epoch

				data = [[], []]
				for idx in range(4):
					original_file = f'{self.checkpoint_dir}/original_{instruments_name[idx]}'
					decoded_file = f'{self.checkpoint_dir}/generated_{instruments_name[idx]}'

					torchaudio.save(uri=original_file,
									src=instruments[idx].unsqueeze(0).detach().cpu(),
									sample_rate=sample_rate)

					torchaudio.save(uri=decoded_file,
									src=output_instruments[idx].unsqueeze(0).detach().cpu(),
									sample_rate=sample_rate)

					data[0].append(wandb.Audio(str(original_file), sample_rate=sample_rate))
					data[1].append(wandb.Audio(str(decoded_file), sample_rate=sample_rate))

				original_full_file = f'{self.checkpoint_dir}/original_full_song.wav'
				decoded_full_file = f'{self.checkpoint_dir}/generated_full_song.wav'

				torchaudio.save(uri=original_full_file, src=mixed.detach().cpu(), sample_rate=sample_rate)
				torchaudio.save(uri=decoded_full_file,
								src=torch.einsum('ij-> j', output_instruments).unsqueeze(0).detach().cpu(),
								sample_rate=sample_rate)

				data[0].append(wandb.Audio(str(original_full_file), sample_rate=sample_rate))
				data[1].append(wandb.Audio(str(decoded_full_file), sample_rate=sample_rate))

				columns = ['bass vs D(bass)', 'drums vs D(drums)', 'guitar vs D(guitar)', 'piano vs D(piano)',
						   'mixed vs D(mixed)']

				self.logger.log_table(key=f'DEMO EPOCH [{epoch}]', columns=columns, data=data)

		except Exception:
			print("CRASHED on_validation_batch_end")
		finally:
			return
