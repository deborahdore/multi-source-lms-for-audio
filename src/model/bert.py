import random
from typing import Any, Tuple

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import wandb
from torch.nn import functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from transformers import BertForMaskedLM, BertTokenizer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class AudioBert(L.LightningModule):
	def __init__(self,
				 learning_rate: float,
				 checkpoint_dir: str,
				 codebook: str,
				 sample_rate: int,
				 frame_length: int,
				 num_embedding: int):
		super(AudioBert, self).__init__()

		self.max_hidden_size = 512
		self.save_hyperparameters("num_embedding", "checkpoint_dir", "sample_rate", "learning_rate")

		self.codebook = torch.tensor(pd.read_csv(codebook).values, requires_grad=False, dtype=torch.float)
		self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
		self.softmax = nn.Softmax(dim=-1)

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		self.mask_token = tokenizer.convert_tokens_to_ids('[MASK]')
		self.pad_token = tokenizer.convert_tokens_to_ids('[PAD]')

		self.conv = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=4, stride=2, padding=1)
		self.linear = nn.Linear(in_features=(sample_rate * frame_length) // 8, out_features=sample_rate * frame_length)

	def forward(self, x: torch.Tensor, batch_size: int, seq_len: int):
		output = []

		if self.training:
			rand_probs = torch.rand(x.size(0), device=x.device)
			x[rand_probs < 0.15] = self.mask_token

		x = x.reshape(batch_size, -1)
		input_size = x.size(1)

		start = 0
		while start < input_size:
			end = start + self.max_hidden_size
			tokens = x[:, start:end]
			attention_mask = torch.ones((tokens.size(0), 512), device=x.device)
			if tokens.size(1) < self.max_hidden_size:
				padding = torch.zeros(batch_size,
									  self.max_hidden_size - tokens.size(1),
									  dtype=tokens.dtype,
									  device=tokens.device)
				attention_mask[:, tokens.size(1):] = 0
				tokens = torch.cat((tokens, padding), dim=1)

			bert_output = self.bert(tokens.to(torch.int), attention_mask=attention_mask).logits
			bert_output = self.softmax(bert_output).argmax(dim=-1)
			output.append(bert_output)
			start = end

		output = torch.cat(output, dim=1)
		output = output[:, :input_size].reshape(-1)
		output = torch.round((output / output.max()) * (self.max_hidden_size - 1))

		encodings = torch.zeros(output.size(0), self.hparams.num_embedding, device=x.device)
		encodings.scatter_(1, output.to(torch.int64).unsqueeze(1), 1)

		quantized = torch.matmul(encodings, self.codebook.to(x.device)).view(batch_size, seq_len // 4, -1)
		quantized = torch.einsum('bwc -> bcw', quantized)

		output = self.linear(self.conv(quantized.to(x.device)))
		return output

	def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int = None):
		quantized, instruments = batch
		output = self.forward(quantized.squeeze(), batch_size=instruments.size(-0), seq_len=instruments.size(-1))
		return output

	def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
		quantized, instruments = batch
		batch_size = instruments.size(0)
		output = self.forward(quantized.squeeze(), batch_size=batch_size, seq_len=instruments.size(-1))

		loss = 0
		for i in range(4):
			loss += F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])

		self.log("train/loss", loss, on_epoch=True, on_step=True, batch_size=batch_size, prog_bar=True)
		return loss

	def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
		return self.calculate_loss(batch, "validation")

	def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
		return self.calculate_loss(batch, "test")

	def calculate_loss(self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str):
		quantized, instruments = batch
		batch_size = instruments.size(0)
		output = self.forward(quantized.squeeze(), batch_size=batch_size, seq_len=instruments.size(-1))
		mixed_output = torch.einsum('bij-> bj', output)
		mixed = torch.einsum('bij-> bj', instruments)
		instruments_name = ["bass", "drums", "guitar", "piano"]
		loss = 0
		for i, instrument in enumerate(instruments_name):
			loss += F.mse_loss(input=output[:, i, :], target=instruments[:, i, :])

			# MSE LOSS
			self.log(f"{mode}/l2_{instrument}_loss",
					 F.mse_loss(input=output[:, i, :], target=instruments[:, i, :]),
					 on_step=False,
					 on_epoch=True,
					 prog_bar=False,
					 batch_size=batch_size)

			# SI_SDR
			self.log(f"{mode}/si_sdr_{instrument}_measure",
					 scale_invariant_signal_distortion_ratio(preds=output[:, i, :], target=instruments[:, i,
																						   :]).mean(),
					 on_step=False,
					 on_epoch=True,
					 prog_bar=False,
					 batch_size=batch_size)
		# SI-SDR
		self.log(f"{mode}/si_sdr_full_audio_measure",
				 scale_invariant_signal_distortion_ratio(preds=mixed_output, target=mixed).mean(),
				 on_epoch=True,
				 on_step=False,
				 prog_bar=False,
				 batch_size=batch_size)
		# MSE loss
		self.log(f"{mode}/l2_full_audio_loss",
				 F.mse_loss(input=mixed_output, target=mixed),
				 on_epoch=True,
				 on_step=False,
				 prog_bar=False,
				 batch_size=batch_size)
		self.log(f"{mode}/loss", loss, on_epoch=True, on_step=True, batch_size=batch_size, prog_bar=True)

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
				quantized, instruments = batch
				batch_size = instruments.size(0)
				index = random.randint(0, batch_size - 1)
				quantized = quantized.reshape(batch_size, -1)[index]
				instruments = instruments[index]

				output = self.forward(quantized.squeeze(), batch_size=1, seq_len=instruments.size(-1))

				output_instruments = output.squeeze()

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

				torchaudio.save(uri=original_full_file,
								src=torch.einsum('ij-> j', instruments).unsqueeze(0).detach().cpu(),
								sample_rate=self.hparams.sample_rate)
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
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
		return optimizer
