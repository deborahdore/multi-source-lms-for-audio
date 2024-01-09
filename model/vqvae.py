from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torchaudio
import wandb
from torch import nn as nn, optim
from torch.nn import functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from model.decoder import Decoder
from model.encoder import Encoder
from model.modules import VectorQuantizer

""" Main model that initializes training/validation/test phases """


class VQVAE(L.LightningModule):
	def __init__(self,
				 num_hidden: int,
				 num_residual_layer: int,
				 num_residual_hidden: int,
				 num_embedding: int,
				 embedding_dim: int,
				 commitment_cost: float,
				 learning_rate: int,
				 checkpoint_dir: str,
				 codebook_file: str):
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
		self.codebook_file = codebook_file
		self.best_loss = float('inf')
		self.weights = [0.25] * 4
		self.matrix_loss_per_instrument = []
		self.save_hyperparameters()

	def training_step(self, batch, batch_idx):
		""" Training step with l2 loss on each instrument """
		mixed, instruments = batch

		# commitment loss + embedding loss
		output, embedding_loss, commitment_loss, perplexity = self.forward(mixed)

		loss = embedding_loss + commitment_loss

		# loss per instrument
		for i in range(4):
			mse_loss = F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])
			# to improve the sound of each instrument
			si_sdr_loss = -1 * scale_invariant_signal_distortion_ratio(preds=output[:, i, :],
																	   target=instruments[:, i, :]).mean()
			loss += mse_loss + self.weights[i] * si_sdr_loss

		self.log("train/loss", loss, on_epoch=True)
		self.log("train/perplexity", perplexity, on_epoch=True)
		return loss

	def validation_step(self, batch, batch_idx):
		""" Validation step that calculates: l1, l2, si-sdr, l1 and l2 on spectrogram and logs them """
		return self.calculate_loss(batch, "validation")

	def test_step(self, batch, batch_idx):
		""" Test step that calculates different measures: l1, l2, si-sdr, l1 and l2 on spectrogram and logs them """
		return self.calculate_loss(batch, "test")

	def forward(self, x):
		""" Forward function that carries out the main operations """
		z = self.conv(self.encoder(x))
		embedding_loss, commitment_loss, quantized, perplexity, encodings = self.vector_quantizer(z)
		output = self.decoder(quantized)
		return output, embedding_loss, commitment_loss, perplexity

	@torch.no_grad()
	def get_quantized(self, x):
		""" Used during inference to retrieve the quantized representation of an input """
		z = self.conv(self.encoder(x))
		embedding_loss, commitment_loss, quantized, perplexity, encodings = self.vector_quantizer(z)
		return quantized, encodings

	def calculate_loss(self, batch, mode: str):
		""" Calculates losses during validation and testing step """
		mixed, instruments = batch
		output, embedding_loss, commitment_loss, perplexity = self.forward(mixed)
		mixed_output = torch.einsum('bij-> bj', output)

		to_spectrogram = torchaudio.transforms.MelSpectrogram(
			sample_rate=self.trainer.val_dataloaders.dataset.target_sample_rate,
			n_fft=400,
			win_length=400,
			hop_length=160,
			n_mels=64).to(mixed.device)

		instruments_name = ["bass", "drums", "guitar", "piano"]

		# commitment loss
		self.log(f"{mode}/embedding_loss", embedding_loss, on_epoch=True)

		# embedding loss
		self.log(f"{mode}/commitment_loss", commitment_loss, on_epoch=True)

		self.log(f"{mode}/perplexity", perplexity, on_epoch=True)

		loss = embedding_loss + commitment_loss

		# loss per instrument
		new_weights_loss_per_instrument = []
		for i, instrument in enumerate(instruments_name):
			mse_loss = F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])
			# to improve the sound of each instrument
			si_sdr_loss = -1 * scale_invariant_signal_distortion_ratio(preds=output[:, i, :],
																	   target=instruments[:, i, :]).mean()

			# calculate total loss per instrument
			instruments_loss = mse_loss + self.weights[i] * si_sdr_loss
			# append calculated loss to list in order to compute new weights after
			new_weights_loss_per_instrument.append(instruments_loss.item())
			# add to total loss
			loss += instruments_loss

			self.log(f"{mode}/l2_{instrument}_loss",
					 F.mse_loss(input=output[:, i, :], target=instruments[:, i, :]),
					 on_epoch=True)

			self.log(f"{mode}/l1_{instrument}_loss",
					 F.l1_loss(input=output[:, i, :], target=instruments[:, i, :]),
					 on_epoch=True)

			self.log(f"{mode}/si_sdr_{instrument}_measure",
					 scale_invariant_signal_distortion_ratio(preds=output[:, i, :], target=instruments[:, i,
																						   :]).mean(),
					 on_epoch=True)

			self.log(f"{mode}/spectrogram_l2_{instrument}_loss",
					 F.mse_loss(input=to_spectrogram(output[:, i, :]), target=to_spectrogram(instruments[:, i, :])),
					 on_epoch=True)

		self.matrix_loss_per_instrument.append(new_weights_loss_per_instrument)

		# SI-SDR loss on combined audio
		self.log(f"{mode}/si_sdr_full_audio_measure",
				 scale_invariant_signal_distortion_ratio(preds=mixed_output, target=mixed.squeeze(1)).mean(),
				 on_epoch=True)

		# MSE loss combined audio
		self.log(f"{mode}/l2_full_audio_loss", F.mse_loss(input=mixed_output, target=mixed.squeeze(1)), on_epoch=True)

		# L1 loss combined audio
		self.log(f"{mode}/l1_full_audio_loss", F.l1_loss(input=mixed_output, target=mixed.squeeze(1)), on_epoch=True)

		self.log(f"{mode}/loss", loss, on_epoch=True)

		return loss

	def configure_optimizers(self):
		""" Configure Adam Optimizer for training """
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, amsgrad=False)
		return optimizer

	def on_validation_batch_end(self, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		""" At the end of the validation step, for each epoch, log an example of input and output of the model """
		try:
			# only log on the first batch of validation
			if batch_idx != 0:
				return

			if outputs.item() < self.best_loss:
				self.best_loss = outputs.item()
				self.log("validation/best_loss", self.best_loss, on_epoch=True)

			if not isinstance(self.logger, L.pytorch.loggers.wandb.WandbLogger):
				return

			instruments_name = ["bass", "drums", "guitar", "piano"]

			with torch.no_grad():
				mixed, instruments = batch
				mixed = mixed[0]
				instruments = instruments[0]
				output = self.forward(mixed.unsqueeze(0))
				output_instruments = output[0].squeeze()

				sample_rate = self.trainer.val_dataloaders.dataset.target_sample_rate
				epoch = self.trainer.current_epoch

				data = [[], []]
				for idx in range(4):
					original_file = f'{self.checkpoint_dir}/original_{instruments_name[idx]}.wav'
					decoded_file = f'{self.checkpoint_dir}/generated_{instruments_name[idx]}.wav'

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

	def on_train_epoch_end(self):
		""" At the end of each epoch save the codebook """
		codebook_weights = self.vector_quantizer.codebook.weight.data.cpu().numpy()
		codebook_dataframe = pd.DataFrame(codebook_weights)
		codebook_dataframe.to_csv(self.codebook_file)

	def on_validation_epoch_end(self):
		""" Calculate new weights """
		mean_losses = torch.einsum("ij -> j",
								   torch.tensor(self.matrix_loss_per_instrument) / len(
									   self.matrix_loss_per_instrument))

		avg = torch.mean(mean_losses)
		self.weights = [0.25 * (loss / avg) for loss in mean_losses]
		self.matrix_loss_per_instrument = []
