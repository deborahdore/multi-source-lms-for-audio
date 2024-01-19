from typing import Optional, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SlakhDataset
from src.data.transform import Quantize
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SlakhDataModule(L.LightningDataModule):
	""" Custom Datamodule for Slakh Dataset"""

	def __init__(self,
				 train_dir: str,
				 val_dir: str,
				 test_dir: str,
				 target_sample_rate: int,
				 target_frame_length_sec: int,
				 batch_size: int,
				 num_workers: int = 1,
				 pin_memory: bool = False,
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
			self.val_dataset = SlakhDataset(self.hparams.val_dir,
											target_sample_rate=self.hparams.target_sample_rate,
											frame_length_sec=self.hparams.target_frame_length_sec)

			log.info(f"Training dataset length: {len(self.train_dataset)}")
			log.info(f"Validation dataset length: {len(self.val_dataset)}")

		if stage == "test" or "predict":
			if stage == "predict" and self.test_dataset is not None:
				return

			self.test_dataset = SlakhDataset(self.hparams.test_dir,
											 target_sample_rate=self.hparams.target_sample_rate,
											 frame_length_sec=self.hparams.target_frame_length_sec)
			log.info(f"Testing dataset length: {len(self.test_dataset)}")

	def train_dataloader(self):
		return DataLoader(self.train_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  drop_last=True,
						  shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  drop_last=True,
						  shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.test_dataset,
						  batch_size=self.hparams.batch_size,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  drop_last=True,
						  shuffle=False)

	def predict_dataloader(self):
		return DataLoader(self.test_dataset,
						  batch_size=1,
						  num_workers=self.hparams.num_workers,
						  pin_memory=self.hparams.pin_memory,
						  shuffle=False)

	def on_after_batch_transfer(self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int):
		mixed, instruments = batch
		if self.transform:
			return self.transform(mixed), instruments
		return mixed, instruments
