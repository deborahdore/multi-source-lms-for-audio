from typing import Optional, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SlakhDataset
from src.data.transform import Quantize
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SlakhDataModule(L.LightningDataModule):

	def __init__(self,
				 train_dir: str,
				 val_dir: str,
				 test_dir: str,
				 target_sample_rate: int,
				 target_sample_duration: int,
				 max_duration: int,
				 maximum_dataset_size: int,
				 batch_size: int,
				 persistent_workers: bool = True,
				 num_workers: int = 1,
				 pin_memory: bool = False,
				 masking: bool = False,
				 quantizer: Optional[Quantize] = None):
		"""
		Custom Datamodule for Slakh

		@param train_dir: path to the training directory
		@param val_dir: path to the validation directory
		@param test_dir: path to the testing directory
		@param target_sample_rate: sample rate at which resample the songs
		@param target_sample_duration: duration in seconds of the samples that will compose the dataloader's batch
		@param max_duration: maximum duration in seconds of each song
		@param maximum_dataset_size: maximum size of the dataset
		@param batch_size: batch size of the dataloader
		@param persistent_workers: retain workers
		@param num_workers: number of workers for each dataloader
		@param pin_memory
		@param masking: apply masking to audio
		@param train_bert: if True, returns encoding idx of codebook as batch first element
		@param train_transformer: if True, returns quantized representation as batch first element
		@param quantizer: object that performs quantization
		"""

		super().__init__()

		self.train_dir = train_dir
		self.val_dir = val_dir
		self.test_dir = test_dir

		self.quantize = quantizer
		self.train_transformer = train_transformer
		self.train_bert = train_bert

		self.target_sample_rate = target_sample_rate
		self.target_sample_duration = target_sample_duration
		self.max_duration = max_duration
		self.maximum_dataset_size = maximum_dataset_size
		self.batch_size = batch_size
		self.pin_memory = pin_memory
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers
		self.masking = masking

	def create_dataset(self, path: str):
		return SlakhDataset(path,
							target_sample_rate=self.target_sample_rate,
							target_sample_duration=self.target_sample_duration,
							max_duration=self.max_duration,
							maximum_dataset_size=self.maximum_dataset_size,
							masking=self.masking)

	def train_dataloader(self):
		## !!! do not create a dataset during setup. Due to a lightning bug, it could downgrade the performances
		return DataLoader(self.create_dataset(self.train_dir),
						  batch_size=self.batch_size,
						  pin_memory=self.pin_memory,
						  num_workers=self.num_workers,
						  persistent_workers=self.persistent_workers,
						  drop_last=True,
						  shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.create_dataset(self.val_dir),
						  batch_size=self.batch_size,
						  num_workers=self.num_workers,
						  pin_memory=self.pin_memory,
						  persistent_workers=self.persistent_workers,
						  drop_last=True,
						  shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.create_dataset(self.test_dir),
						  batch_size=self.batch_size,
						  num_workers=self.num_workers,
						  pin_memory=self.pin_memory,
						  persistent_workers=self.persistent_workers,
						  drop_last=True,
						  shuffle=False)

	def predict_dataloader(self):
		return DataLoader(self.create_dataset(self.test_dir),
						  batch_size=1,
						  num_workers=self.num_workers,
						  pin_memory=self.pin_memory,
						  persistent_workers=self.persistent_workers,
						  shuffle=False)

	def on_after_batch_transfer(self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int):
		if self.quantize:
			# return self.quantize.get_quantized(batch), batch
			return self.quantize.get_encodings_idx(batch), batch

		# train vqvae
		mixture_frame = torch.einsum('ij->j', batch)
		return torch.stack([mixture_frame] * 4, dim=0), batch
