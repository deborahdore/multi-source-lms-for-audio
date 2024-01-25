import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.models import vgg16

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PerceptualLoss(nn.Module):
	def __init__(self, sample_rate: int):
		super(PerceptualLoss, self).__init__()
		vgg16_model = vgg16(weights='DEFAULT')
		vgg16_model.eval()

		for param in vgg16_model.parameters():
			param.requires_grad = False

		self.features = vgg16_model.features
		self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
																n_fft=400,
																win_length=400,
																hop_length=160,
																n_mels=64)

	def forward(self, x: torch.Tensor, target: torch.Tensor):
		with torch.no_grad():
			log.info(x.shape)
			log.info(target.shape)

			spectrogram_input = self.spectrogram(x)
			spectrogram_target = self.spectrogram(target)

			spectrogram_input = torch.cat([spectrogram_input.unsqueeze(1)] * 3, dim=1)
			spectrogram_target = torch.cat([spectrogram_target.unsqueeze(1)] * 3, dim=1)

			features_input = self.features(spectrogram_input)
			features_target = self.features(spectrogram_target)

			return F.mse_loss(input=features_input, target=features_target)