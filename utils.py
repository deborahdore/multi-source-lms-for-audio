from __future__ import print_function

import lightning as L
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def generate_sound(config: DictConfig, model: L.LightningModule, dataloader: DataLoader):
	model.eval()
	instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
	mixture, instruments = dataloader
	output, _ = model(mixture)
	output = output.squeeze().detach()
	instruments = instruments.squeeze().detach()

	for idx in range(instruments.size(0)):
		torchaudio.save(uri=f'{config.path.output_dir}/real_{instruments_name[idx]}',
						src=instruments[idx].unsqueeze(0),
						sample_rate=22050)

		torchaudio.save(uri=f'{config.path.output_dir}/fake_{instruments_name[idx]}',
						src=output[idx].unsqueeze(0),
						sample_rate=22050)
