import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset


class SlakhDataset(Dataset):
	def __init__(self, data_dir: str, frame_length_sec: int = 3, target_sample_rate: int = 16000):
		self.data_dir = data_dir
		self.frame_length_sec = frame_length_sec
		self.target_sample_rate = target_sample_rate

		self.file_paths = []
		self.instruments = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
		self.frame_length_samples = int(frame_length_sec * self.target_sample_rate)

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
				# 3 seconds of audio at least
				if length >= self.frame_length_sec:
					new_file_paths.append(self.file_paths[idx])
		self.file_paths = new_file_paths

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
		return torch.stack(instrument_data)

	def __len__(self):
		return len(self.file_paths) * 100

	def resample(self, audio, original_freq):
		return torchaudio.functional.resample(audio, orig_freq=original_freq, new_freq=self.target_sample_rate)

	def __getitem__(self, idx: int):
		idx = idx % len(self.file_paths)

		instruments = self.get_stems(idx)
		mixture = sum(instruments)

		while True:
			offset = random.randint(0, (int(mixture.size(1) / self.frame_length_samples) - 1))

			mixture_offset = mixture[:, offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]
			instruments_offset = instruments[:, :,
								 offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]

			if sum(mixture_offset.squeeze()) > 0:
				break

		return torch.Tensor(mixture_offset), torch.Tensor(instruments_offset.squeeze(1))

# if __name__ == '__main__':
# 	data = SlakhDataset("/Users/deborahdore/Documents/multi-source-lms-for-audio/slakh2100/test")
# 	next(iter(data))
