import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchaudio
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from model.vqvae import VQVAE


def plot_codebook(config: DictConfig):
	codebook_df = pd.read_csv(config.path.codebook_file)
	codebook_weights = codebook_df.values

	kmeans = KMeans(n_clusters=4, random_state=42)
	clusters = kmeans.fit_predict(codebook_weights)

	sns.set(style='whitegrid')
	plt.figure(figsize=(8, 6))
	sns.scatterplot(x=codebook_weights[:, 0], y=codebook_weights[:, 1], hue=clusters, palette='viridis', legend='full')
	plt.title('Codebook Embeddings - KMeans Clustering (k=4)')
	plt.legend(title='Codebook Embeddings Clusters')
	plt.show()


def plot_embeddings_from_quantized(config: DictConfig, input: torch.Tensor):
	codebook_df = pd.read_csv(config.path.codebook_file)
	codebook_weights = codebook_df.values

	kmeans = KMeans(n_clusters=4, random_state=42)
	clusters = kmeans.fit_predict(codebook_weights)

	model = VQVAE.load_from_checkpoint(f"{config.path.checkpoint_dir}/last.ckpt")
	model.eval()
	_, encodings = model.get_quantized(input)
	embeddings = torch.matmul(encodings, torch.Tensor(codebook_df.values))

	sns.set(style='whitegrid')
	plt.figure(figsize=(8, 6))
	sns.scatterplot(x=codebook_weights[:, 0], y=codebook_weights[:, 1], hue=clusters, palette='viridis', legend='full')
	sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], palette='viridis')
	plt.title('Embeddings from Quantized Representation')
	plt.show()


def plot_waveform(waveform: torch.Tensor, plot_dir:str, sample_rate: int = 22050, title:str = None):
	# Calculate the time axis for the waveform
	total_samples = waveform.shape[1]
	time_axis = torch.arange(0, total_samples) / sample_rate

	# Plot the waveform
	plt.figure(figsize=(10, 4))
	plt.plot(time_axis.numpy(), waveform.t().numpy())
	plt.xlabel('Time (s)')
	plt.ylabel('Amplitude')
	plt.title(f'{title} Waveform')
	plt.grid(True)
	# plt.show()
	plt.savefig(f"{plot_dir}/{title}_waveform.svg")

def plot_spectrogram(waveform: torch.Tensor, plot_dir:str, sample_rate: int = 22050, title:str = None):
	# Compute the Mel spectrogram
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
														   n_fft=400,
														   win_length=400,
														   hop_length=160,
														   n_mels=128)(waveform)

	# Convert power spectrogram to dB scale
	mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

	# Plot the Mel spectrogram
	plt.figure(figsize=(10, 4))
	plt.imshow(mel_spectrogram_db[0].detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
	plt.xlabel('Time')
	plt.ylabel('Mel Filterbanks')
	plt.title(f'{title} Spectrogram')
	plt.colorbar(format='%+2.0f dB')
	# plt.show()
	plt.savefig(f"{plot_dir}/{title}_spectrogram.svg")