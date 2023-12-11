from __future__ import print_function

import os
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import SlakhDataset
from model import VQVAE


def init(config: DictConfig):
	OmegaConf.register_new_resolver("base_dir", lambda x: os.path.abspath("."))

	assert Path(config.path.train_dir).exists()
	assert Path(config.path.test_dir).exists()
	assert Path(config.path.val_dir).exists()

	Path(config.path.logging_dir).mkdir(parents=True, exist_ok=True)
	Path(config.path.output_dir).mkdir(parents=True, exist_ok=True)
	if config.logger.wandb:
		Path(config.path.logging_dir + "/wandb").mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
	init(config)

	train = DataLoader(SlakhDataset(config.path.train_dir,
									target_sample_rate=config.data.target_sample_rate,
									frame_length_sec=config.data.target_frame_length_sec),
					   batch_size=config.trainer.batch_size,
					   shuffle=True)

	val = DataLoader(SlakhDataset(config.path.val_dir,
								  target_sample_rate=config.data.target_sample_rate,
								  frame_length_sec=config.data.target_frame_length_sec),
					 batch_size=config.trainer.batch_size,
					 shuffle=False)

	test = DataLoader(SlakhDataset(config.path.test_dir,
								   target_sample_rate=config.data.target_sample_rate,
								   frame_length_sec=config.data.target_frame_length_sec),
					  batch_size=config.trainer.batch_size,
					  shuffle=False)

	model = VQVAE(num_hidden=config.model.num_hidden,
				  num_residual_layer=config.model.num_residual_layer,
				  num_residual_hidden=config.model.num_residual_hidden,
				  num_embedding=config.model.num_embeddings,
				  embedding_dim=config.model.embedding_dim,
				  commitment_cost=config.model.commitment_cost,
				  learning_rate=config.model.learning_rate,
				  output_dir=config.path.output_dir)

	if config.logger.wandb:
		logger = WandbLogger(project=config.logger.wandb_project_name,
							 save_dir=config.path.logging_dir,
							 offline=config.logger.wandb_offline)
	else:
		logger = TensorBoardLogger(save_dir=config.path.logging_dir)

	trainer = L.Trainer(enable_checkpointing=False,
						# uncomment next 4 rows for debugging
						# fast_dev_run=True,
						# accelerator="cpu",
						# strategy="ddp",
						# devices=1,
						max_epochs=config.trainer.max_epochs,
						callbacks=[EarlyStopping(monitor=config.trainer.early_stopping_monitor,
												 mode=config.trainer.early_stopping_monitor_mode,
												 patience=config.trainer.early_stopping_patience)],
						profiler=config.trainer.profiler,
						logger=logger)

	trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
	model.eval()
	trainer.test(model=model, dataloaders=test)


if __name__ == '__main__':
	main()
