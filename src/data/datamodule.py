import gc
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
				 do_clean: bool = True,
				 transform: Optional[Quantize] = None):
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
		@param transform: optional transformation to apply to batch before training
		"""

		super().__init__()

		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None

		self.transform = transform
		self.save_hyperparameters(logger=False)

	def setup(self, stage: str = None):

		if stage == "fit":
			if self.train_dataset is None:
				self.train_dataset = self.create_dataset(self.hparams.train_dir)
			if self.val_dataset is None:
				self.val_dataset = self.create_dataset(self.hparams.val_dir)
		else:
			if self.test_dataset is None:
				self.test_dataset = self.create_dataset(self.hparams.test_dir)

	def create_dataset(self, path: str):
		return SlakhDataset(path,
							target_sample_rate=self.hparams.target_sample_rate,
							target_sample_duration=self.hparams.target_sample_duration,
							max_duration=self.hparams.max_duration,
							maximum_dataset_size=self.hparams.maximum_dataset_size,
							do_clean=self.hparams.do_clean)

	def train_dataloader(self):
		if self.train_dataset is None:
			self.train_dataset = self.create_dataset(self.hparams.train_dir)
		return DataLoader(self.train_dataset,
						  batch_size=self.hparams.batch_size,
						  pin_memory=self.hparams.pin_memory,
						  num_workers=self.hparams.num_workers,
						  persistent_workers=self.hparams.persistent_workers,
						  drop_last=True,
						  shuffle=True)

	def val_dataloader(self):
		if self.val_dataset is None:
			self.val_dataset = self.create_dataset(self.hparams.val_dir)
		return DataLoader(self.val_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  persistent_workers=self.hparams.persistent_workers,
						  drop_last=True,
						  shuffle=False)

	def test_dataloader(self):
		if self.test_dataset is None:
			self.test_dataset = self.create_dataset(self.hparams.test_dir)
		return DataLoader(self.test_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  persistent_workers=self.hparams.persistent_workers,
						  drop_last=True,
						  shuffle=False)

	def predict_dataloader(self):
		if self.test_dataset is None:
			self.test_dataset = self.create_dataset(self.hparams.test_dir)
		return DataLoader(self.test_dataset,
						  batch_size=1,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  persistent_workers=self.hparams.persistent_workers,
						  shuffle=False)

	def on_after_batch_transfer(self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int):
		mixed, instruments = batch
		if self.transform:
			return self.transform(mixed), instruments
		return mixed, instruments
