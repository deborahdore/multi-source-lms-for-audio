import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset


class SlakhDataset(Dataset):
	def __init__(self, data_dir: str, frame_length_sec: int = 3, sample_rate: int = 22050):
		self.data_dir = data_dir
		self.frame_length_sec = frame_length_sec
		self.file_paths = []
		self.instruments = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
		self.sample_rate = sample_rate
		self.frame_length_samples = int(frame_length_sec * self.sample_rate)

		for sub_dir in [x for x in os.walk(data_dir)][0][1]:
			self.file_paths.append(os.path.join(data_dir, sub_dir))

		self.clean_df()

	def clean_df(self):
		new_file_paths = []
		for idx in range(0, len(self.file_paths)):
			files = [files for root, _, files in os.walk(self.file_paths[idx])]
			first_instrument_in_directory = os.path.join(self.file_paths[idx], files[0][0])
			if first_instrument_in_directory:
				info = torchaudio.info(first_instrument_in_directory)
				length = info.num_frames / info.sample_rate
				if length >= self.frame_length_sec:  # 3 seconds of audio at least
					new_file_paths.append(self.file_paths[idx])

		self.file_paths = new_file_paths

	def get_stems(self, idx: int):
		instrument_data = []

		for instrument in self.instruments:
			file_path = os.path.join(self.file_paths[idx], instrument)
			if os.path.exists(file_path):
				audio, _ = torchaudio.load(file_path)
				instrument_data.append(audio)
			else:
				instrument_data.append(torch.zeros(1, 1))  # If file doesn't exist, append zeros

		max_len = max(audio.shape[-1] for audio in instrument_data)
		# Pad or truncate to have the same length for all instruments
		instrument_data = [torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1])) for audio in instrument_data]
		return torch.stack(instrument_data)

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, idx):
		instruments = self.get_stems(idx)
		mixture = sum(instruments)

		offset = random.randint(0, (int(mixture.size(1) / self.frame_length_samples) - 1))

		mixture_offset = mixture[:, offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]
		instruments_offset = instruments[:, :,
							 offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples].squeeze(1)

		return torch.Tensor(mixture_offset), torch.Tensor(instruments_offset)
