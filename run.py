import os
from pathlib import Path

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from data import SlakhDataModule
from model import VQVAE

torch.set_float32_matmul_precision('medium')


def init(config: DictConfig):
	OmegaConf.register_new_resolver("base_dir", lambda x: os.path.abspath("."))

	assert Path(config.path.train_dir).exists()
	assert Path(config.path.test_dir).exists()
	assert Path(config.path.val_dir).exists()

	Path(config.path.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
	try:
		init(config)

		data_module = SlakhDataModule(config)

		model = VQVAE(num_hidden=config.model.num_hidden,
					  num_residual_layer=config.model.num_residual_layer,
					  num_residual_hidden=config.model.num_residual_hidden,
					  num_embedding=config.model.num_embeddings,
					  embedding_dim=config.model.embedding_dim,
					  commitment_cost=config.model.commitment_cost,
					  learning_rate=config.model.learning_rate,
					  checkpoint_dir=config.path.checkpoint_dir,
					  batch_size=config.trainer.batch_size)

		if config.logger.wandb:
			wandb.finish()
			logger = WandbLogger(name=config.logger.wandb_name,
								 project=config.logger.wandb_project_name,
								 save_dir=config.path.checkpoint_dir,
								 log_model=False,
								 offline=config.logger.wandb_offline,
								 settings=wandb.Settings(init_timeout=300))
		else:
			logger = TensorBoardLogger(save_dir=config.path.checkpoint_dir)

		trainer = L.Trainer(max_epochs=config.trainer.max_epochs,
							default_root_dir=config.path.checkpoint_dir,
							enable_checkpointing=False,
							enable_progress_bar=True,
							callbacks=[EarlyStopping(monitor=config.trainer.early_stopping_monitor,
													 mode=config.trainer.early_stopping_monitor_mode,
													 patience=config.trainer.early_stopping_patience)],
							profiler=config.trainer.profiler,
							logger=logger,
							log_every_n_steps=None,
							# uncomment next 4 rows for debugging
							# fast_dev_run=True,
							# accelerator="cpu",
							# strategy="ddp",
							# devices=1,
							)

		trainer.fit(model=model, datamodule=data_module)
		trainer.test(model=model, datamodule=data_module)

	finally:
		wandb.finish()


if __name__ == '__main__':
	main()
