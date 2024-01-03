import torch
from torch import nn as nn


class TransformerDecoder(nn.Module):
	def __init__(self, input_dim=12000, output_dim=48000, num_layers=4, num_heads=8, hidden_dim=512):
		super(TransformerDecoder, self).__init__()

		self.output_dim = output_dim
		self.hidden_dim = hidden_dim

		self.embedding = nn.Linear(input_dim, hidden_dim)
		self.positional_encoding = PositionalEncoding(hidden_dim)
		decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
		self.fc = nn.Linear((hidden_dim * 64) // 4, output_dim)

	def forward(self, x):
		# Assuming input shape: (batch_size, sequence_length, input_dim)
		batch_size = x.size(0)
		seq_len = x.size(1)

		# Reshape the input to (sequence_length, batch_size, input_dim)
		x = x.permute(1, 0, 2)

		# Embedding the input
		x = self.embedding(x)
		x = self.positional_encoding(x)

		# Transformer decoder
		tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
		memory = torch.zeros(seq_len, batch_size, self.hidden_dim).to(x.device)  # Initialize memory

		output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)

		# Transpose output to (batch_size, sequence_length, output_dim)
		output = output.permute(1, 0, 2).reshape(batch_size, 4, -1)

		# Fully connected layer for output
		output = self.fc(output)

		return output


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=10000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return x + self.pe[:, :x.size(1)].detach()
