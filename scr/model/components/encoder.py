from torch import nn as nn
from torch.nn import functional as F

from scr.model.components.residualstack import ResidualStack


class Encoder(nn.Module):
	def __init__(self, in_channel: int, num_hidden: int, num_residual_layer: int, num_residual_hidden: int):
		super(Encoder, self).__init__()

		self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=num_hidden // 2, kernel_size=4, stride=2,
							   padding=1)

		self.conv2 = nn.Conv1d(in_channels=num_hidden // 2, out_channels=num_hidden, kernel_size=4, stride=2,
							   padding=1)

		self.conv3 = nn.Conv1d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

		self.residual_stack = ResidualStack(in_channel=num_hidden,
											num_hidden=num_hidden,
											num_residual_layer=num_residual_layer,
											num_residual_hidden=num_residual_hidden)

	def forward(self, x):
		# initial dimension: BCW
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.conv3(x)
		return self.residual_stack(x)
