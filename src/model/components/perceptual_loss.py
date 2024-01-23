import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
	def __init__(self):
		super(PerceptualLoss, self).__init__()
		# Load pre-trained VGG16
		vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)

		# Modify the first layer to accept 1 channel of 1D audio data
		vgg16.features[0] = nn.Conv1d(1, 64, kernel_size=(3,), stride=(1,), padding=(1,))

		# Remove max-pooling layers
		vgg16.features = vgg16.features[:-3]

		# Freeze the parameters
		for param in vgg16.parameters():
			param.requires_grad = False

		# Extract only the feature extraction part of the VGG16
		self.features = vgg16.features

	def forward(self, reconstructed, real):
		# Extract features from the reconstructed and real signals
		features_reconstructed = self.features(reconstructed)
		features_real = self.features(real)

		# Calculate Mean Squared Error (MSE) loss between the extracted features
		loss = F.mse_loss(features_reconstructed, features_real)

		return loss
