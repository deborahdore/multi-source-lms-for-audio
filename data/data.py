import pytorch_lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dataset import SlakhDataset


class SlakhDataModule(L.LightningDataModule):
	def __init__(self, config: DictConfig, transform=None):
		super().__init__()
		self.test_dataset = None
		self.train_dataset = None
		self.val_dataset = None
		self.config = config
		self.transform = transform

	def setup(self, stage=None):
		if stage in ["fit", None]:
			self.train_dataset = SlakhDataset(self.config.path.train_dir,
											  target_sample_rate=self.config.data.target_sample_rate,
											  frame_length_sec=self.config.data.target_frame_length_sec)

			self.val_dataset = SlakhDataset(self.config.path.val_dir,
											target_sample_rate=self.config.data.target_sample_rate,
											frame_length_sec=self.config.data.target_frame_length_sec)

			print(f"[setup] train dataset len: {len(self.train_dataset)}")
			print(f"[setup] val dataset len: {len(self.val_dataset)}")

		if stage in ["test", "predict", None]:
			self.test_dataset = SlakhDataset(self.config.path.test_dir,
											 target_sample_rate=self.config.data.target_sample_rate,
											 frame_length_sec=self.config.data.target_frame_length_sec)
			print(f"[setup] test dataset len: {len(self.test_dataset)}")

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.config.trainer.batch_size, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.config.trainer.batch_size, shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.config.trainer.batch_size, shuffle=False)

	def on_after_batch_transfer(self, batch, dataloader_idx: int):
		if self.transform:
			mixed, instruments = batch
			return self.transform(mixed), instruments
