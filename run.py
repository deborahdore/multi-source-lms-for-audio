import os
from pathlib import Path

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from data import SlakhDataModule
from model.vqvae import VQVAE

torch.set_float32_matmul_precision('medium')


def init(config: DictConfig):
	OmegaConf.register_new_resolver("base_dir", lambda x: os.path.abspath("."))

	assert Path(config.path.train_dir).exists()
	assert Path(config.path.test_dir).exists()
	assert Path(config.path.val_dir).exists()

	Path(config.path.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
	init(config)

	data_module = SlakhDataModule(config)

	model = VQVAE(num_hidden=config.model.num_hidden,
				  num_residual_layer=config.model.num_residual_layer,
				  num_residual_hidden=config.model.num_residual_hidden,
				  num_embedding=config.model.num_embeddings,
				  embedding_dim=config.model.embedding_dim,
				  commitment_cost=config.model.commitment_cost,
				  learning_rate=config.model.learning_rate,
				  checkpoint_dir=config.path.checkpoint_dir)
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
										  filename='best_model',
										  save_last=True)

	early_stopping = EarlyStopping(monitor=config.trainer.early_stopping_monitor,
								   mode=config.trainer.early_stopping_monitor_mode,
								   patience=config.trainer.early_stopping_patience)

	trainer = L.Trainer(max_epochs=config.trainer.max_epochs,
						default_root_dir=config.path.checkpoint_dir,
						enable_progress_bar=True,
						callbacks=[early_stopping, checkpoint_callback],
						profiler=config.trainer.profiler,
						logger=logger,
						log_every_n_steps=None,
						accelerator="gpu",
						# uncomment next rows for debugging
						# fast_dev_run=True,
						# accelerator="cpu",
						# devices=1
						)

	checkpoint_path = None
	if config.trainer.load_from_checkpoint:
		checkpoint_path = f"{config.path.checkpoint_dir}/last.ckpt"

	trainer.fit(model=model, datamodule=data_module, ckpt_path=checkpoint_path)
	trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
	main()
