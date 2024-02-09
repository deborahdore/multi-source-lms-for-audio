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
				 do_masking: bool = False,
				 intra_source: bool = False,
				 inter_source: bool = False,
				 probability: float = 0.2,
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
		@param transform: optional transformation to apply to batch before training
		"""

		super().__init__()

		self.train_dir = train_dir
		self.val_dir = val_dir
		self.test_dir = test_dir

		self.quantize = quantizer

		self.target_sample_rate = target_sample_rate
		self.target_sample_duration = target_sample_duration
		self.max_duration = max_duration
		self.maximum_dataset_size = maximum_dataset_size
		self.batch_size = batch_size
		self.pin_memory = pin_memory
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers
		self.do_masking = do_masking
		self.intra_source = intra_source
		self.inter_source = inter_source
		self.probability = probability

	def create_dataset(self, path: str, do_masking: bool = False):
		if do_masking:
			return SlakhDataset(path,
								target_sample_rate=self.target_sample_rate,
								target_sample_duration=self.target_sample_duration,
								max_duration=self.max_duration,
								maximum_dataset_size=self.maximum_dataset_size,
								do_masking=self.do_masking,
								intra_source=self.intra_source,
								inter_source=self.inter_source,
								probability=self.probability)

		return SlakhDataset(path,
							target_sample_rate=self.target_sample_rate,
							target_sample_duration=self.target_sample_duration,
							max_duration=self.max_duration,
							maximum_dataset_size=self.maximum_dataset_size)

	def train_dataloader(self):
		## !!! do not create a dataset during setup. Due to a lightning bug, it could downgrade the performances
		return DataLoader(self.create_dataset(self.train_dir, do_masking=self.do_masking),
						  batch_size=self.batch_size,
						  pin_memory=self.pin_memory,
						  num_workers=self.num_workers,
						  persistent_workers=self.persistent_workers,
						  drop_last=True,
						  shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.create_dataset(self.val_dir, do_masking=False),
						  batch_size=self.batch_size,
						  num_workers=self.num_workers,
						  pin_memory=self.pin_memory,
						  persistent_workers=self.persistent_workers,
						  drop_last=True,
						  shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.create_dataset(self.test_dir, do_masking=False),
						  batch_size=self.batch_size,
						  num_workers=self.num_workers,
						  pin_memory=self.pin_memory,
						  persistent_workers=self.persistent_workers,
						  drop_last=True,
						  shuffle=False)

	def predict_dataloader(self):
		return DataLoader(self.create_dataset(self.test_dir, do_masking=False),
						  batch_size=1,
						  num_workers=self.num_workers,
						  pin_memory=self.pin_memory,
						  persistent_workers=self.persistent_workers,
						  shuffle=False)

	def on_after_batch_transfer(self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int):
		mixed, instruments = batch

		if self.quantize:
			return mixed, instruments, self.quantize(mixed)
		return mixed, instruments
