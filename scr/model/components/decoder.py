from torch import nn as nn
from torch.nn import functional as F

from scr.model.components.residual_stack import ResidualStack


class Decoder(nn.Module):
	def __init__(self, in_channel: int, num_hidden: int, num_residual_layer: int, num_residual_hidden: int):
		super(Decoder, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

		self.residual_stack = ResidualStack(in_channel=num_hidden,
											num_hidden=num_hidden,
											num_residual_layer=num_residual_layer,
											num_residual_hidden=num_residual_hidden)

		self.conv1_transpose = nn.ConvTranspose1d(in_channels=num_hidden,
												  out_channels=num_hidden // 2,
												  kernel_size=4,
												  stride=2,
												  padding=1)

		self.conv2_transpose = nn.ConvTranspose1d(in_channels=num_hidden // 2,
												  out_channels=4,
												  kernel_size=4,
												  stride=2,
												  padding=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.residual_stack(x)
		x = F.relu(self.conv1_transpose(x))
		return self.conv2_transpose(x)
