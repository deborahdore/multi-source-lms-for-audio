import os
import random

import torch
import torchaudio

from scr.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SlakhDataset(torch.utils.data.Dataset):
	""" Custom Dataset for Slakh """

	def __init__(self, data_dir: str, frame_length_sec: int, target_sample_rate: int):

		self.data_dir = data_dir
		self.frame_length_sec = frame_length_sec
		self.target_sample_rate = target_sample_rate

		self.file_paths = []
		self.frame_length_samples = int(frame_length_sec * self.target_sample_rate)
		for sub_dir in [x for x in os.walk(data_dir)][0][1]:
			self.file_paths.append(os.path.join(data_dir, sub_dir))

		self.clean_df()

		# save in memory to train faster
		self.instruments_dict = {}
		for idx in range(len(self.file_paths)):
			self.instruments_dict[idx] = self.get_stems(idx)

	def clean_df(self):
		""" Remove songs that only contain silence or that are less than 3 seconds"""
		log.info(f"Dataset cleaning: {self.data_dir}")

		new_file_paths = []
		for idx in range(0, len(self.file_paths)):
			instruments = self.get_stems(idx)
			is_silence = int(torch.einsum('ij-> ', instruments))
			if is_silence <= 0:
				continue
			length = instruments.size(-1) // self.target_sample_rate
			if length < 3:
				continue
			new_file_paths.append(self.file_paths[idx])

		self.file_paths = new_file_paths

	def get_stems(self, idx: int):
		""" Load instruments from folder and stack them into a 4xN matrix"""
		instrument_data = []

		for instrument in ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]:
			file_path = os.path.join(self.file_paths[idx], instrument)
			if os.path.exists(file_path):
				audio, sample_rate = torchaudio.load(file_path)
				audio = self.resample(audio, sample_rate)
				instrument_data.append(audio)
			else:
				# if the instrument is not in the song, append zeros
				instrument_data.append(torch.zeros(1, 1))

		# Pad or truncate to have the same length for all instruments
		max_len = max(audio.shape[-1] for audio in instrument_data)
		instrument_data = [torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1])) for audio in instrument_data]

		return torch.stack(instrument_data).squeeze()

	def __len__(self):
		""" Avoid wasting songs with this workaround,
		otherwise the model will iterate over n second of the song at each epoch """

		augmented_length = len(self.file_paths) * 1000
		return augmented_length

	def resample(self, audio, original_freq):
		""" Resample audio to insure every song is sampled at the same frequency"""
		return torchaudio.functional.resample(audio, orig_freq=original_freq, new_freq=self.target_sample_rate)

	def __getitem__(self, idx: int):
		instruments = self.instruments_dict[idx % len(self.file_paths)]  # workaround to get more frames
		mixture = torch.einsum('ij-> j', instruments).unsqueeze(0)

		while True:
			# choose random frame from song
			offset = random.randint(0, (mixture.size(1) // self.frame_length_samples) - 1)

			mixture_offset = mixture[:, offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]
			instruments_offset = instruments[:,
								 offset * self.frame_length_samples: (offset + 1) * self.frame_length_samples]

			if torch.einsum('ij->', mixture_offset) > 0:  # avoid silence only
				break

		return mixture_offset, instruments_offset
