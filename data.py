import os
import random

import lightning as L
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class SlakhDataset(Dataset):
	def __init__(self, data_dir: str, frame_length_sec: int, target_sample_rate: int):

		self.data_dir = data_dir
		self.frame_length_sec = frame_length_sec
		self.target_sample_rate = target_sample_rate

		self.file_paths = []
		self.instruments = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
		self.frame_length_samples = int(frame_length_sec * self.target_sample_rate)

		for sub_dir in [x for x in os.walk(data_dir)][0][1]:
			self.file_paths.append(os.path.join(data_dir, sub_dir))

		self.clean_df()

		# accelerate training
		self.instruments_dict = {}
		for idx in range(len(self.file_paths)):
			self.instruments_dict[idx] = self.get_stems(idx)

	def clean_df(self):
		print("[clean_df] Starting dataset cleaning")
		new_file_paths = []
		for idx in range(0, len(self.file_paths)):
			instruments = self.get_stems(idx)
			is_silence = int(torch.einsum('ij-> ', instruments))
			if is_silence <= 0:
				continue
			length = instruments.size(-1) // self.target_sample_rate
			if length < 5:
				continue
			new_file_paths.append(self.file_paths[idx])

		self.file_paths = new_file_paths
		print("[clean_df] Finished dataset cleaning")

	def get_stems(self, idx: int):
		instrument_data = []

		for instrument in self.instruments:
			file_path = os.path.join(self.file_paths[idx], instrument)
			if os.path.exists(file_path):
				audio, sample_rate = torchaudio.load(file_path)
				audio = self.resample(audio, sample_rate)
				instrument_data.append(audio)
			else:
				instrument_data.append(torch.zeros(1, 1))  # If file doesn't exist, append zeros

		max_len = max(audio.shape[-1] for audio in instrument_data)
		# Pad or truncate to have the same length for all instruments
		instrument_data = [torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1])) for audio in instrument_data]
		return torch.stack(instrument_data).squeeze()

	def __len__(self):
		augmented_length = len(self.file_paths) * 100
		return augmented_length

	def resample(self, audio, original_freq):
		return torchaudio.functional.resample(audio, orig_freq=original_freq, new_freq=self.target_sample_rate)

	def __getitem__(self, idx: int):
		instruments = self.instruments_dict[idx % len(self.file_paths)]
		mixture = torch.einsum('ij-> j', instruments).unsqueeze(0)

		while True:
			offset = random.randint(0, (mixture.size(1) // self.frame_length_samples) - 1)

			mixture_offset = mixture[:, offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]
			instruments_offset = instruments[:,
								 offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]

			if torch.einsum('ij->', mixture_offset) > 0:
				break

		return mixture_offset, instruments_offset


class SlakhDataModule(L.LightningDataModule):
	def __init__(self, config: DictConfig):
		super().__init__()
		self.test_dataset = None
		self.train_dataset = None
		self.val_dataset = None
		self.config = config

	def setup(self, stage=None):
		self.train_dataset = SlakhDataset(self.config.path.train_dir,
										  target_sample_rate=self.config.data.target_sample_rate,
										  frame_length_sec=self.config.data.target_frame_length_sec)

		self.val_dataset = SlakhDataset(self.config.path.val_dir,
										target_sample_rate=self.config.data.target_sample_rate,
										frame_length_sec=self.config.data.target_frame_length_sec)

		self.test_dataset = SlakhDataset(self.config.path.test_dir,
										 target_sample_rate=self.config.data.target_sample_rate,
										 frame_length_sec=self.config.data.target_frame_length_sec)

		print(f"[setup] train dataset len: {len(self.train_dataset)}")
		print(f"[setup] test dataset len: {len(self.test_dataset)}")
		print(f"[setup] val dataset len: {len(self.val_dataset)}")

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.config.trainer.batch_size, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.config.trainer.batch_size, shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.config.trainer.batch_size, shuffle=False)

# if __name__ == '__main__':
# 	data = SlakhDataset("/Users/deborahdore/Documents/multi-source-lms-for-audio/slakh2100/test")
# 	next(iter(data))
