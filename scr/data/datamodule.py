from typing import Optional, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from scr.data.dataset import SlakhDataset
from scr.data.transform import Quantize

from scr.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SlakhDataModule(L.LightningDataModule):
	""" Custom Datamodule for Slakh Dataset"""
	def __init__(self,
				 train_dir: str,
				 val_dir: str,
				 test_dir: str,
				 target_sample_rate: int,
				 target_frame_length_sec: int,
				 batch_size: int,
				 num_workers: int,
				 pin_memory: bool,
				 transform: Optional[Quantize] = None):

		super().__init__()

		self.test_dataset = None
		self.train_dataset = None
		self.val_dataset = None

		self.transform = transform
		self.save_hyperparameters(logger=False)

	def setup(self, stage: Optional[str] = None):
		"""
		Lightning calls this method before `trainer.fit()`, `trainer.validate()`, `trainer.test()`,
		and `trainer.predict()`

		:param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
		"""
		log.info(f"Setting up stage {stage}")

		if stage == 'fit':
			self.train_dataset = SlakhDataset(self.hparams.train_dir,
											  target_sample_rate=self.hparams.target_sample_rate,
											  frame_length_sec=self.hparams.target_frame_length_sec)

		if stage == 'validate':
			self.val_dataset = SlakhDataset(self.hparams.val_dir,
											target_sample_rate=self.hparams.target_sample_rate,
											frame_length_sec=self.hparams.target_frame_length_sec)

		if stage == "test" or "predict":
			if stage == "predict" and self.test_dataset is not None:
				return

			self.test_dataset = SlakhDataset(self.hparams.test_dir,
											 target_sample_rate=self.hparams.target_sample_rate,
											 frame_length_sec=self.hparams.target_frame_length_sec)

		print(f"[setup] train dataset len: {len(self.train_dataset)}")
		print(f"[setup] val dataset len: {len(self.val_dataset)}")
		print(f"[setup] test dataset len: {len(self.test_dataset)}")

	def train_dataloader(self):
		return DataLoader(self.train_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.test_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  shuffle=False)

	def predict_dataloader(self):
		return DataLoader(self.test_dataset,
						  batch_size=1,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  shuffle=False)

	def on_after_batch_transfer(self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int):
		if self.transform:
			mixed, instruments = batch
			return self.transform(mixed), instruments
