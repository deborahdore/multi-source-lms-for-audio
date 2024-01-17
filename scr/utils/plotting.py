from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchaudio
import umap
from lightning import LightningModule
from omegaconf import DictConfig
from sklearn.cluster import KMeans


def plot_codebook(cfg: DictConfig):
	codebook_df = pd.read_csv(cfg.paths.codebook_file)
	proj = umap.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine', random_state=14).fit_transform(codebook_df.values)
	kmeans = KMeans(n_clusters=4, random_state=14)
	clusters = kmeans.fit_predict(proj)

	sns.set(style='whitegrid')
	plt.figure(figsize=(8, 6))
	sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=clusters, legend='full')
	plt.title('Codebook Embeddings - KMeans Clustering (k=4)')
	plt.legend(title='Clusters')
	plt.savefig(f"{cfg.paths.plot_dir}/codebook.svg")


def plot_embeddings_from_quantized(cfg: DictConfig, batch: Tuple[torch.Tensor, torch.Tensor]):
	codebook_df = pd.read_csv(cfg.paths.codebook_file)
	proj = umap.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine', random_state=14).fit_transform(codebook_df.values)
	kmeans = KMeans(n_clusters=4, random_state=14)
	clusters = kmeans.fit_predict(proj)

	instruments_name = ["bass", "drums", "guitar", "piano"]

	checkpoint = torch.load(f"{cfg.paths.checkpoint_dir}/best_model.ckpt", map_location=torch.device('cpu'))
	vqvae: LightningModule = hydra.utils.instantiate(cfg.model)
	vqvae.load_state_dict(checkpoint['state_dict'])
	vqvae.eval()

	mixed, instruments = batch
	for idx in range(instruments.size(1)):
		one_instrument = instruments[:, idx, :].unsqueeze(0)
		quantized, encodings, encodings_indices = vqvae.get_quantized(one_instrument)
		encodings_indices = torch.unique(encodings_indices)
		selected_embeddings = proj[encodings_indices]

		sns.set(style='whitegrid')
		plt.figure(figsize=(8, 6))
		sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=clusters, legend='full')
		sns.scatterplot(x=selected_embeddings[:, 0], y=selected_embeddings[:, 1], alpha=0.5, color='yellow')
		plt.title(f'{instruments_name[idx].upper()} Embeddings')
		plt.legend(title='Clusters')
		plt.savefig(f"{cfg.paths.plot_dir}/{instruments_name[idx].lower()}_embeddings_quantized_representation.svg")


def plot_waveform(waveform: torch.Tensor, plot_dir: str, sample_rate: int = 22050, title: str = None):
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


def plot_spectrogram(waveform: torch.Tensor, plot_dir: str, sample_rate: int = 22050, title: str = None):
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
