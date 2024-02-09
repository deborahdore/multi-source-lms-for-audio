import json
import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SlakhDataset(Dataset):
	def __init__(self,
				 data_dir: str,
				 target_sample_duration: int,
				 target_sample_rate: int,
				 max_duration: int,
				 maximum_dataset_size: int,
				 do_masking: bool = False,
				 intra_source: bool = False,
				 inter_source: bool = False,
				 probability: float = 0.2):
		"""
		Custom Dataset for Slakh

		@param data_dir: path to the dataset directory
		@param target_sample_rate: sample rate at which resample the songs
		@param target_sample_duration: duration in seconds of the samples that will compose the dataloader's batch
		@param max_duration: maximum duration in seconds of each song
		"""

		self.data_dir = data_dir  # dataset directory
		self.save_file = os.path.join(data_dir, "dataset_dict.json")
		self.target_sample_duration = target_sample_duration  # duration in seconds of each sample
		self.target_sample_rate = target_sample_rate  # sampling rate
		self.max_duration = max_duration  # maximum duration of a song
		self.maximum_dataset_size = maximum_dataset_size  # maximum size of the dataset
		self.do_masking = do_masking
		self.intra_source = intra_source
		self.inter_source = inter_source
		self.probability = probability

		# load file paths

		self.file_paths = []
		for sub_dir in [x for x in os.walk(self.data_dir)][0][1]:
			self.file_paths.append(os.path.join(self.data_dir, sub_dir))

		self.data_list = []
		if not os.path.isfile(self.save_file):
			self.clean_and_load()

		self.data_list = json.load(open(self.save_file))
		self.data_dict = {}
		self.load()

	def load(self):
		dict_idx = -1
		for elem in self.data_list:
			new_idx = elem.get('file_path_idx')
			if dict_idx == new_idx: continue
			dict_idx = new_idx
			self.data_dict[dict_idx] = torch.load(f"{self.data_dir}/tensor_{dict_idx}.pt")

	def clean_and_load(self):
		"""
			Clean dataset:
			1. resample
			2. cut songs at 2 minutes
			3. remove songs that only contain one instrument
			4. delete tracks with only silence
			5. save to a tensor
			6. for each song:
				6.1 split song into frames, each having a fixed duration
				6.2 check that the frame doesn't contain only silence
				6.3 drop last incomplete frame
				6.4 save a dict with frame start and end for each frame
		"""
		log.info(f"Dataset cleaning: {self.data_dir}")
		assert self.save_file is not None

		file_paths = []
		for idx in range(len(self.file_paths)):

			instruments, num_of_instruments = self.get_stems(idx)

			if num_of_instruments < 2:
				log.info(f"Track {self.file_paths[idx]} with only one instrument")
				continue

			if int(torch.einsum('ij->', instruments)) == 0:
				log.info(f"Track {self.file_paths[idx]} with only silence")
				continue

			file_paths.append(self.file_paths[idx])
			torch.save(instruments, f'{self.data_dir}/tensor_{idx}.pt')

			for sub_idx in range(0, self.max_duration):
				frame_start = sub_idx * self.target_sample_rate
				frame_end = (sub_idx + self.target_sample_duration) * self.target_sample_rate
				instruments_frame = instruments[:, frame_start:frame_end]

				if int(torch.einsum('ij->', instruments_frame)) == 0:
					continue
				if instruments_frame.shape[-1] != self.target_sample_rate * self.target_sample_duration:
					# drop last incomplete
					continue

				self.data_list.append({'file_path_idx': idx, 'frame_start': frame_start, 'frame_end': frame_end})

		self.file_paths = file_paths
		with open(self.save_file, "w") as file:
			json.dump(self.data_list, file)

		log.info(f"Finished dataset cleaning: {self.data_dir}")

	def get_stems(self, idx: int):
		""" Load instruments from folder, resample, trim and stack them into a 4xN matrix"""
		instrument_data = []
		num_of_instruments = 0

		for instrument in ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]:
			file_path = os.path.join(self.file_paths[idx], instrument)
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
		return len(self.data_list)

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

	def masking(self, x: torch.Tensor):
		if self.intra_source:
			num_non_zero_rows = (x.sum(dim=1) != 0).sum().item()  # total numer of instruments
			max_instruments_to_mask = min(num_non_zero_rows - 2, 3)  # always leave 2 instruments unmasked at least

			if max_instruments_to_mask > 0:
				# randomly select how many instruments to mask
				num_instruments_to_mask = torch.randint(1, max_instruments_to_mask + 1, (1,))
				# randomly select rows to mask
				rows_to_mask = torch.randperm(num_non_zero_rows)[:num_instruments_to_mask]

				for row in rows_to_mask:
					x[row.item(), :] = 0

		if self.inter_source:
			# Inter-source masking: apply dropout across the entire input tensor
			x = F.dropout(x, p=self.probability, training=True)

		return x

	def __getitem__(self, idx: int):
		elem_dict = self.data_list[idx]
		instruments = self.data_dict[elem_dict.get('file_path_idx')]
		instruments_frame = instruments[:, elem_dict.get('frame_start'):elem_dict.get('frame_end')]

		if self.do_masking:
			mixture_frame_masked = torch.einsum('ij->j', self.masking(instruments_frame)).unsqueeze(0)
			return mixture_frame_masked, instruments_frame

		mixture_frame = torch.einsum('ij->j', instruments_frame).unsqueeze(0)
		return mixture_frame, instruments_frame
