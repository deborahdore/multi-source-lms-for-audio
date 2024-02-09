import os.path
from typing import List

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# necessary before importing any local modules e.g. `from src import utils`
# finds the file .project-root and sets its position as root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.plotting import plot_codebook, plot_embeddings_from_quantized, plot_spectrogram, plot_waveform
from src.utils.util import extras, get_metric_value, task_wrapper
from src.data.transform import Quantize

torch.set_float32_matmul_precision("medium")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@task_wrapper
def train_vqvae(cfg: DictConfig):
	data_module: LightningDataModule = hydra.utils.instantiate(cfg.data)

	vqvae: LightningModule = hydra.utils.instantiate(cfg.model.vqvae)
	vqvae.to(device)
	logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

	callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

	trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

	object_dict = {
		"cfg"       : cfg,
		"datamodule": data_module,
		"model"     : vqvae,
		"callbacks" : callbacks,
		"logger"    : logger,
		"trainer"   : trainer, }

	if cfg.train:
		trainer.fit(model=vqvae, datamodule=data_module, ckpt_path=cfg.ckpt_path)
	train_metrics = trainer.callback_metrics

	if cfg.test:
		trainer.test(model=vqvae, datamodule=data_module, ckpt_path=cfg.ckpt_path)
	test_metrics = trainer.callback_metrics

	metric_dict = {**train_metrics, **test_metrics}

	return metric_dict, object_dict


@task_wrapper
def train_transformer(cfg: DictConfig):
	vqvae: LightningModule = hydra.utils.instantiate(cfg.model.vqvae)
	best_vqvae_file = f"{cfg.paths.best_checkpoint_dir}/best_vqvae.ckpt"
	assert os.path.exists(best_vqvae_file)
	state_dict = torch.load(best_vqvae_file, map_location=device)['state_dict']
	vqvae.load_state_dict(state_dict)
	vqvae.to(device)
	vqvae.eval()

	quantizer: Quantize = Quantize(vqvae)

	data_module: LightningDataModule = hydra.utils.instantiate(cfg.data, quantizer=quantizer)

	transformer: LightningModule = hydra.utils.instantiate(cfg.model.transformer)
	transformer.to(device)

	logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

	callbacks = None
	if "callbacks" in cfg.keys() and cfg.callbacks is not None:
		model_checkpoint_callback: Callback = hydra.utils.instantiate(cfg.callbacks.model_checkpoint,
																	  filename='best_transformer')

		early_stopping_callback: Callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)

		callbacks = [model_checkpoint_callback, early_stopping_callback]

	trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

	object_dict = {
		"cfg"       : cfg,
		"datamodule": data_module,
		"model"     : vqvae,
		"callbacks" : callbacks,
		"logger"    : logger,
		"trainer"   : trainer, }

	if cfg.train:
		trainer.fit(model=transformer, datamodule=data_module, ckpt_path=cfg.ckpt_path)
	train_metrics = trainer.callback_metrics

	if cfg.test:
		trainer.test(model=transformer, datamodule=data_module, ckpt_path=cfg.ckpt_path)
	test_metrics = trainer.callback_metrics

	metric_dict = {**train_metrics, **test_metrics}

	return metric_dict, object_dict


def visualize(cfg: DictConfig):
	data_module: LightningDataModule = hydra.utils.instantiate(cfg.data, batch_size=1)
	data_module.setup(stage='predict')

	instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
	mixed, instruments = next(iter(data_module.predict_dataloader()))

	plot_embeddings_from_quantized(cfg, batch=(mixed, instruments), device=device)
	plot_codebook(cfg)

	for idx, instrument_name in enumerate(instruments_name):
		plot_spectrogram(instruments[:, idx, :], plot_dir=cfg.paths.plot_dir, title=instrument_name.split(".")[0])
		plot_waveform(instruments[:, idx, :], plot_dir=cfg.paths.plot_dir, title=instrument_name.split(".")[0])

	plot_spectrogram(mixed.squeeze(0), plot_dir=cfg.paths.plot_dir, title="song")
	plot_waveform(mixed.squeeze(0), plot_dir=cfg.paths.plot_dir, title="song")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
	extras(cfg)
	if cfg.get("seed"):
		L.seed_everything(cfg.seed, workers=True)

	metric_dict = {}
	if cfg.train_vqvae:
		metric_dict, _ = train_vqvae(cfg)

	if cfg.train_transformer:
		metric_dict, _ = train_transformer(cfg)

	# visualize(cfg)

	# safely retrieve metric value for hydra-based hyperparameter optimization
	metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

	# return optimized metric for optuna
	return metric_value


if __name__ == '__main__':
	main()
