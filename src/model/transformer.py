import random
from typing import Any

import lightning as L
import torch
import torchaudio
import wandb
from torch import nn as nn, optim
from torch.nn import functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TransformerQuantizerDecoder(L.LightningModule):
	def __init__(self,
				 sample_rate: int,
				 frame_length: int,
				 learning_rate: float,
				 checkpoint_dir: str,
				 num_layers: int = 4,
				 num_heads: int = 8,
				 hidden_dim: int = 512):
		super(TransformerQuantizerDecoder, self).__init__()

		self.save_hyperparameters()

		output_dim = sample_rate * frame_length
		input_dim = (sample_rate * frame_length) // 4

		self.embedding = nn.Linear(input_dim, hidden_dim)
		self.positional_encoding = PositionalEncoding(hidden_dim)
		decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
		self.fc = nn.Linear((hidden_dim * 64) // 4, output_dim)

	def training_step(self, batch, batch_idx):
		mixed, instruments, quantized = batch
		output = self.forward(quantized)

		# loss per instrument
		loss = 0
		for i in range(4):
			loss += F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])

		self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		return self.calculate_loss(batch, mode="validation")

	def test_step(self, batch, batch_idx):
		return self.calculate_loss(batch, mode="testing")

	def forward(self, x):
		# Assuming input shape: (batch_size, sequence_length, input_dim)
		batch_size = x.size(0)
		seq_len = x.size(1)

		# Reshape the input to (sequence_length, batch_size, input_dim)
		x = x.permute(1, 0, 2)

		# Embedding the input
		x = self.embedding(x)
		x = self.positional_encoding(x)

		# Transformer decoder
		tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
		memory = torch.zeros(seq_len, batch_size, self.hparams.hidden_dim).to(x.device)  # Initialize memory

		output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)

		# Transpose output to (batch_size, sequence_length, output_dim)
		output = output.permute(1, 0, 2).reshape(batch_size, 4, -1)

		# Fully connected layer for output
		output = self.fc(output)

		return output

	def calculate_loss(self, batch, mode: str):
		""" Calculates losses during a validation and testing step """
		mixed, instruments, quantized = batch
		output = self.forward(quantized)
		mixed_output = torch.einsum('bij-> bj', output)

		instruments_name = ["bass", "drums", "guitar", "piano"]

		# loss per instrument
		loss = 0
		for i, instrument in enumerate(instruments_name):
			instruments_loss = F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])
			loss += instruments_loss

			# MSE LOSS
			self.log(f"{mode}/l2_{instrument}_loss", instruments_loss, on_step=False, on_epoch=True, prog_bar=False)

			# L1 LOSS
			self.log(f"{mode}/l1_{instrument}_loss",
					 F.l1_loss(input=output[:, i, :], target=instruments[:, i, :]),
					 on_step=False,
					 on_epoch=True,
					 prog_bar=False)
			# SI_SDR
			self.log(f"{mode}/si_sdr_{instrument}_measure",
					 scale_invariant_signal_distortion_ratio(preds=output[:, i, :], target=instruments[:, i,
																						   :]).mean(),
					 on_step=False,
					 on_epoch=True,
					 prog_bar=False)

		# SI-SDR
		self.log(f"{mode}/si_sdr_full_audio_measure",
				 scale_invariant_signal_distortion_ratio(preds=mixed_output, target=mixed.squeeze(1)).mean(),
				 on_epoch=True,
				 on_step=False,
				 prog_bar=False)

		# MSE loss
		self.log(f"{mode}/l2_full_audio_loss",
				 F.mse_loss(input=mixed_output, target=mixed.squeeze(1)),
				 on_epoch=True,
				 on_step=False,
				 prog_bar=False)

		# L1 loss
		self.log(f"{mode}/l1_full_audio_loss",
				 F.l1_loss(input=mixed_output, target=mixed.squeeze(1)),
				 on_epoch=True,
				 on_step=False,
				 prog_bar=False)

		self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def on_validation_batch_end(self, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		""" At the end of the validation step, for each epoch, log an example of input and output of the model """
		try:
			# only log on the first batch of validation
			if batch_idx != 0:
				return

			if not isinstance(self.logger, L.pytorch.loggers.wandb.WandbLogger):
				return

			instruments_name = ["bass", "drums", "guitar", "piano"]

			with torch.no_grad():
				mixed, instruments, quantized = batch
				index = random.randint(0, mixed.size(0)-1)
				mixed = mixed[index]
				instruments = instruments[index]
				quantized = quantized[index]

				output = self.forward(quantized.unsqueeze(0))
				output_instruments = output[0].squeeze()

				epoch = self.trainer.current_epoch

				data = [[], []]
				for idx in range(4):
					original_file = f'{self.hparams.checkpoint_dir}/original_{instruments_name[idx]}.wav'
					decoded_file = f'{self.hparams.checkpoint_dir}/generated_{instruments_name[idx]}.wav'

					torchaudio.save(uri=original_file,
									src=instruments[idx].unsqueeze(0).detach().cpu(),
									sample_rate=self.hparams.sample_rate)

					torchaudio.save(uri=decoded_file,
									src=output_instruments[idx].unsqueeze(0).detach().cpu(),
									sample_rate=self.hparams.sample_rate)

					data[0].append(wandb.Audio(str(original_file), sample_rate=self.hparams.sample_rate))
					data[1].append(wandb.Audio(str(decoded_file), sample_rate=self.hparams.sample_rate))

				original_full_file = f'{self.hparams.checkpoint_dir}/original_full_song.wav'
				decoded_full_file = f'{self.hparams.checkpoint_dir}/generated_full_song.wav'

				torchaudio.save(uri=original_full_file, src=mixed.detach().cpu(), sample_rate=self.hparams.sample_rate)
				torchaudio.save(uri=decoded_full_file,
								src=torch.einsum('ij-> j', output_instruments).unsqueeze(0).detach().cpu(),
								sample_rate=self.hparams.sample_rate)

				data[0].append(wandb.Audio(str(original_full_file), sample_rate=self.hparams.sample_rate))
				data[1].append(wandb.Audio(str(decoded_full_file), sample_rate=self.hparams.sample_rate))

				columns = ['bass vs D(bass)', 'drums vs D(drums)', 'guitar vs D(guitar)', 'piano vs D(piano)',
						   'mixed vs D(mixed)']

				self.logger.log_table(key=f'DEMO EPOCH [{epoch}]', columns=columns, data=data)

		except Exception as err:
			log.warning("Exception while executing -on validation batch end- during transformer training")
			log.warning(err)
		finally:
			return

	def configure_optimizers(self):
		""" Configure Adam Optimizer for training """
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, amsgrad=False)
		return optimizer


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, max_len: int = 10000):
		super(PositionalEncoding, self).__init__()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return x + self.pe[:, :x.size(1)].detach()
