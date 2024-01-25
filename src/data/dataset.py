import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger
from os import listdir
from os.path import isfile, join

log = RankedLogger(__name__, rank_zero_only=True)


class SlakhDataset(Dataset):
	def __init__(self,
				 data_dir: str,
				 target_sample_duration: int,
				 target_sample_rate: int,
				 max_duration: int,
				 maximum_dataset_size: int,
				 do_clean: bool = True):
		"""
		Custom Dataset for Slakh

		@param data_dir: path to the dataset directory
		@param target_sample_rate: sample rate at which resample the songs
		@param target_sample_duration: duration in seconds of the samples that will compose the dataloader's batch
		@param max_duration: maximum duration in seconds of each song
		"""

		self.data_dir = data_dir  # dataset directory
		self.target_sample_duration = target_sample_duration  # duration in seconds of each sample
		self.target_sample_rate = target_sample_rate  # sampling rate
		self.max_duration = max_duration  # maximum duration of a song
		self.maximum_dataset_size = maximum_dataset_size  # maximum size of the dataset

		# load file paths

		self.old_file_paths = []
		for sub_dir in [x for x in os.walk(self.data_dir)][0][1]:
			self.old_file_paths.append(os.path.join(self.data_dir, sub_dir))

		# clean and save dataset in memory
		if do_clean:
			self.clean_and_save()

		self.new_file_paths = [join(f"{data_dir}_new", f) for f in listdir(f"{data_dir}_new") if
							   isfile(os.path.join(f"{data_dir}_new", f)) and f.endswith(".wav")]

	def clean_and_save(self):
		"""
			Clean dataset:
			1. resample
			2. cut songs at 2 minutes
			3. remove songs that only contain one instrument
			4. delete tracks with only silence
			5. for each song:
				5.1 split song into frames, each having a fixed duration
				5.2 check that the frame doesn't contain only silence
				5.3 save in memory
		"""
		log.info(f"Dataset cleaning: {self.data_dir}")
		Path(f"{self.data_dir}_new").mkdir(exist_ok=True, parents=True)

		file_paths = []
		dict_idx = 0
		for idx in range(len(self.old_file_paths)):

			if dict_idx > self.maximum_dataset_size: break

			instruments, num_of_instruments = self.get_stems(idx)

			if num_of_instruments < 2:
				log.info(f"Track {self.old_file_paths[idx]} with only one instrument")
				continue

			if int(torch.einsum('ij->', instruments)) == 0:
				log.info(f"Track {self.old_file_paths[idx]} with only silence")
				continue

			file_paths.append(self.old_file_paths[idx])

			for sub_idx in range(0, self.max_duration):
				frame_start = sub_idx * self.target_sample_rate
				frame_end = (sub_idx + self.target_sample_duration) * self.target_sample_rate
				instruments_frame = instruments[:, frame_start:frame_end]

				if int(torch.einsum('ij->', instruments_frame)) == 0:
					continue
				if instruments_frame.shape[-1] != self.target_sample_rate * self.target_sample_duration:
					# drop last incomplete
					continue

				torchaudio.save(uri=f"{self.data_dir}_new/Track{idx}_{sub_idx}.wav",
								src=instruments.squeeze(),
								sample_rate=self.target_sample_rate)
				dict_idx += 1

		self.old_file_paths = file_paths

	def get_stems(self, idx: int):
		""" Load instruments from folder, resample, trim and stack them into a 4xN matrix"""
		instrument_data = []
		num_of_instruments = 0

		for instrument in ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]:
			file_path = os.path.join(self.old_file_paths[idx], instrument)
			if os.path.exists(file_path):
				audio, sample_rate = torchaudio.load(file_path)
				audio = self.resample(audio, sample_rate)
				audio = self.cut(audio)
				instrument_data.append(audio)
				num_of_instruments += 1
			else:
				# if the instrument is not in the song, append zeros
				instrument_data.append(torch.zeros(1, 1))

		# Pad or truncate to have the same length for all instruments
		max_len = max(audio.shape[-1] for audio in instrument_data)
		instrument_data = [torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1])) for audio in instrument_data]

		return torch.stack(instrument_data).squeeze(), num_of_instruments

	def __len__(self):
		return len(self.new_file_paths)

	def resample(self, audio, original_freq):
		""" Resample audio to insure every song is sampled at the same frequency"""
		return torchaudio.functional.resample(audio, orig_freq=original_freq, new_freq=self.target_sample_rate)

	def cut(self, song: torch.Tensor, trim: int = 10):
		""" Cut songs that last more than 2 minutes and get most central part to retrieve most important information"""
		song = song[:, int(self.target_sample_rate * trim):-int(self.target_sample_rate * trim)]
		song_duration = song.size(1) // self.target_sample_rate
		if song_duration > self.max_duration:
			return song[:, :int(self.max_duration * self.target_sample_rate)]
		else:
			new_duration = (song_duration // self.target_sample_duration) * self.target_sample_duration
			return song[:, :int(new_duration * self.target_sample_rate)]

	def __getitem__(self, idx: int):
		instruments, _ = torchaudio.load(self.new_file_paths[idx])
		mixture = torch.einsum('ij->j', instruments).unsqueeze(0)
		return mixture, instruments
