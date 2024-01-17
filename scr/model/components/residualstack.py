from torch import nn as nn
from torch.nn import functional as F


class ResidualStack(nn.Module):
	def __init__(self, in_channel: int, num_hidden: int, num_residual_layer: int, num_residual_hidden: int):
		super(ResidualStack, self).__init__()

		self.residual_layers = nn.ModuleList([nn.Sequential(nn.ReLU(True),
															nn.Conv1d(in_channels=in_channel if i == 0 else num_hidden,
																	  out_channels=num_residual_hidden,
																	  kernel_size=3,
																	  stride=1,
																	  padding=1,
																	  bias=False),
															nn.ReLU(True),
															nn.Conv1d(in_channels=num_residual_hidden,
																	  out_channels=num_hidden,
																	  kernel_size=1,
																	  stride=1,
																	  bias=False)) for i in range(num_residual_layer)])

	def forward(self, x):
		for layer in self.residual_layers:
			x = x + layer(x)
		return F.relu(x)
