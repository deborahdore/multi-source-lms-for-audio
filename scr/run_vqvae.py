from typing import List

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from scr.utils.instantiators import instantiate_callbacks, instantiate_loggers
from scr.utils.plotting import plot_codebook, plot_embeddings_from_quantized, plot_spectrogram, plot_waveform

torch.set_float32_matmul_precision('medium')


def train(cfg: DictConfig):
	if cfg.get("seed"):
		L.seed_everything(cfg.seed, workers=True)

	data_module: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

	vqvae: LightningModule = hydra.utils.instantiate(cfg.model)

	logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

	callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

	trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

	if cfg.train:
		trainer.fit(model=vqvae, datamodule=data_module, ckpt_path=cfg.ckpt_path)

	if cfg.test:
		trainer.test(model=vqvae, datamodule=data_module, ckpt_path=cfg.ckpt_path)


def visualize(cfg: DictConfig):
	data_module: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, batch_size=1)

	instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
	mixed, instruments = next(iter(data_module.predict_dataloader()))

	plot_embeddings_from_quantized(cfg, batch=(mixed, instruments))
	plot_codebook(cfg)

	for idx, instrument_name in enumerate(instruments_name):
		plot_spectrogram(instruments[:, idx, :], plot_dir=cfg.paths.plot_dir, title=instrument_name.split(".")[0])
		plot_waveform(instruments[:, idx, :], plot_dir=cfg.paths.plot_dir, title=instrument_name.split(".")[0])

	plot_spectrogram(mixed.squeeze(0), plot_dir=cfg.paths.plot_dir, title="song")
	plot_waveform(mixed.squeeze(0), plot_dir=cfg.paths.plot_dir, title="song")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
	# OmegaConf.register_new_resolver("eval", eval)

	train(cfg)
	visualize(cfg)


if __name__ == '__main__':
	main()
