import os
from functools import partial
from pathlib import Path

import hydra
import pytorch_lightning as L
import ray
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch.utils.data import DataLoader

from data.data import SlakhDataModule
from data.dataset import SlakhDataset
from model.vqvae import VQVAE
from utils import plot_codebook, plot_embeddings_from_quantized, plot_spectrogram, plot_waveform

torch.set_float32_matmul_precision('medium')


def init(config: DictConfig):
	OmegaConf.register_new_resolver("base_dir", lambda x: os.path.abspath("."))

	assert Path(config.path.train_dir).exists()
	assert Path(config.path.test_dir).exists()
	assert Path(config.path.val_dir).exists()

	Path(config.path.checkpoint_dir).mkdir(parents=True, exist_ok=True)
	Path(config.path.plot_dir).mkdir(parents=True, exist_ok=True)


def train(config: DictConfig):
	data_module = SlakhDataModule(config)

	model = VQVAE(num_hidden=config.model.vqvae.num_hidden,
				  num_residual_layer=config.model.vqvae.num_residual_layer,
				  num_residual_hidden=config.model.vqvae.num_residual_hidden,
				  num_embedding=config.model.vqvae.num_embeddings,
				  embedding_dim=config.model.vqvae.embedding_dim,
				  commitment_cost=config.model.vqvae.commitment_cost,
				  learning_rate=config.model.vqvae.learning_rate,
				  checkpoint_dir=config.path.checkpoint_dir,
				  codebook_file=config.path.codebook_file)

	if config.logger.wandb:
		wandb.finish()
		logger = WandbLogger(name=config.logger.wandb_name,
							 project=config.logger.wandb_project_name,
							 save_dir=config.path.checkpoint_dir,
							 log_model=False,
							 offline=config.logger.wandb_offline,
							 settings=wandb.Settings(init_timeout=300),
							 magic=True,
							 version=config.logger.version if config.trainer.load_from_checkpoint else None,
							 resume='must' if config.trainer.load_from_checkpoint else None)
	else:
		logger = TensorBoardLogger(save_dir=config.path.checkpoint_dir)

	checkpoint_callback = ModelCheckpoint(dirpath=f"{config.path.checkpoint_dir}/",
										  monitor='validation/loss',
										  mode='min',
										  save_top_k=2,
										  filename='best_model',
										  save_last=True)

	early_stopping = EarlyStopping(monitor=config.trainer.early_stopping_monitor,
								   mode=config.trainer.early_stopping_monitor_mode,
								   patience=config.trainer.early_stopping_patience)

	trainer = L.Trainer(max_epochs=config.trainer.max_epochs,
						min_epochs=config.trainer.min_epochs,
						default_root_dir=config.path.checkpoint_dir,
						enable_progress_bar=True,
						callbacks=[early_stopping, checkpoint_callback],
						profiler=config.trainer.profiler,
						logger=logger,
						log_every_n_steps=None,
						# accelerator="gpu",
						# uncomment next rows for debugging
						accelerator="cpu",
						fast_dev_run=True,
						devices=1)

	checkpoint_path = None
	if config.trainer.load_from_checkpoint:
		checkpoint_path = f"{config.path.checkpoint_dir}/best_model.ckpt"

	trainer.fit(model=model, datamodule=data_module, ckpt_path=checkpoint_path)
	trainer.test(model=model, datamodule=data_module, ckpt_path=f"{config.path.checkpoint_dir}/best_model.ckpt")


def wrapper_train_ray(config, hydra_config: DictConfig):
	""" Wrapper for train_ray that sets chosen hyperparameters"""
	OmegaConf.register_new_resolver("base_dir", lambda x: os.path.abspath("."))

	def train_ray(hydra_config: DictConfig):
		data_module = SlakhDataModule(hydra_config)

		model = VQVAE(num_hidden=hydra_config.model.vqvae.num_hidden,
					  num_residual_layer=hydra_config.model.vqvae.num_residual_layer,
					  num_residual_hidden=hydra_config.model.vqvae.num_residual_hidden,
					  num_embedding=hydra_config.model.vqvae.num_embeddings,
					  embedding_dim=hydra_config.model.vqvae.embedding_dim,
					  commitment_cost=hydra_config.model.vqvae.commitment_cost,
					  learning_rate=hydra_config.model.vqvae.learning_rate,
					  checkpoint_dir=hydra_config.path.checkpoint_dir,
					  codebook_file=hydra_config.path.codebook_file)

		logger = TensorBoardLogger(save_dir=hydra_config.path.checkpoint_dir)
		tune_report = TuneReportCheckpointCallback({"loss": "validation/loss"}, on="validation_end")

		early_stopping = EarlyStopping(monitor=hydra_config.trainer.early_stopping_monitor,
									   mode=hydra_config.trainer.early_stopping_monitor_mode,
									   patience=hydra_config.trainer.early_stopping_patience)

		trainer = L.Trainer(default_root_dir=hydra_config.path.checkpoint_dir,
							enable_progress_bar=False,
							enable_checkpointing=False,
							callbacks=[early_stopping, tune_report],
							logger=logger,
							log_every_n_steps=None,
							accelerator="gpu")

		trainer.fit(model=model, datamodule=data_module)

	hydra_config.model.vqvae.num_hidden = config['num_hidden']
	hydra_config.model.vqvae.num_residual_hidden = config['num_residual_hidden']
	hydra_config.model.vqvae.num_residual_layer = config['num_residual_layer']
	hydra_config.model.vqvae.embedding_dim = config['embedding_dim']
	hydra_config.model.vqvae.num_embeddings = config['num_embeddings']
	hydra_config.model.vqvae.learning_rate = config['learning_rate']
	hydra_config.trainer.batch_size = config['batch_size']
	hydra_config.logger.wandb = False
	hydra_config.trainer.load_from_checkpoint = False

	train_ray(hydra_config)


def visualize(config: DictConfig):
	dataset = SlakhDataset(config.path.test_dir,
						   frame_length_sec=config.data.target_frame_length_sec,
						   target_sample_rate=config.data.target_sample_rate)

	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

	instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
	mixed, instruments = next(iter(dataloader))

	plot_embeddings_from_quantized(config, batch=(mixed, instruments), plot_dir=config.path.plot_dir)
	plot_codebook(config, plot_dir=config.path.plot_dir)

	for idx, instrument_name in enumerate(instruments_name):
		plot_spectrogram(instruments[:, idx, :], plot_dir=config.path.plot_dir, title=instrument_name.split(".")[0])
		plot_waveform(instruments[:, idx, :], plot_dir=config.path.plot_dir, title=instrument_name.split(".")[0])

	plot_spectrogram(mixed.squeeze(0), plot_dir=config.path.plot_dir, title="song")
	plot_waveform(mixed.squeeze(0), plot_dir=config.path.plot_dir, title="song")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(hydra_config: DictConfig):
	init(hydra_config)

	if bool(hydra_config.trainer.optimize):
		tune_config = {
			"num_hidden"         : tune.choice([32, 64, 128]),
			"num_residual_layer" : tune.choice([2, 3]),
			"num_residual_hidden": tune.choice([16, 32, 64]),
			"embedding_dim"      : tune.choice([64, 128]),
			"num_embeddings"     : tune.choice([256, 512]),
			"learning_rate"      : tune.loguniform(1e-4, 1e-1),
			"batch_size"         : tune.choice([32, 64, 128])}

		ray.init()
		results = tune.run(partial(wrapper_train_ray, hydra_config=hydra_config),
						   config=tune_config,
						   num_samples=15,
						   # resources_per_trial={'gpu': 1},
						   local_dir=hydra_config.path.checkpoint_dir)

		best_trial = results.get_best_trial("loss", "min", "last")
		print(f"Best trial config: {best_trial.config}")

	else:
		train(hydra_config)
		visualize(hydra_config)


if __name__ == '__main__':
	main()
