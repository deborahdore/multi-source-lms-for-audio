import os
from pathlib import Path

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from data.data import SlakhDataModule
from data.transform import Quantize
from model.transformer import TransformerDecoder
from model.vqvae import VQVAE

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def init(config: DictConfig):
	OmegaConf.register_new_resolver("base_dir", lambda x: os.path.abspath("."))
	OmegaConf.register_new_resolver("eval", eval)

	assert Path(config.path.train_dir).exists()
	assert Path(config.path.test_dir).exists()
	assert Path(config.path.val_dir).exists()

	Path(config.path.checkpoint_dir).mkdir(parents=True, exist_ok=True)
	Path(config.path.plot_dir).mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(config: DictConfig):
	init(config)

	vqvae = VQVAE(num_hidden=config.model.vqvae.num_hidden,
				  num_residual_layer=config.model.vqvae.num_residual_layer,
				  num_residual_hidden=config.model.vqvae.num_residual_hidden,
				  num_embedding=config.model.vqvae.num_embeddings,
				  embedding_dim=config.model.vqvae.embedding_dim,
				  commitment_cost=config.model.vqvae.commitment_cost,
				  learning_rate=config.model.vqvae.learning_rate,
				  checkpoint_dir=config.path.checkpoint_dir,
				  codebook_file=config.path.codebook_file)

	vqvae_file = f"{config.path.checkpoint_dir}/best_model.ckpt"
	assert os.path.isfile(vqvae_file)

	# load and place in evaluation mode
	vqvae.load_state_dict(torch.load(vqvae_file, map_location=device)['state_dict'])
	vqvae.eval()

	data_module = SlakhDataModule(config, transform=Quantize(vqvae))

	# create transformer
	transformer = TransformerDecoder(input_dim=config.model.transformer.input_dim,
									 output_dim=config.model.transformer.output_dim,
									 learning_rate=config.model.transformer.learning_rate,
									 num_layers=config.model.transformer.num_layers,
									 num_heads=config.model.transformer.num_heads,
									 hidden_dim=config.model.transformer.hidden_dim)
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
										  filename='best_model_transformer',
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
		checkpoint_path = f"{config.path.checkpoint_dir}/best_model_transformer.ckpt"

	trainer.fit(model=transformer, datamodule=data_module, ckpt_path=checkpoint_path)
	trainer.test(model=transformer,
				 datamodule=data_module,
				 ckpt_path=f"{config.path.checkpoint_dir}/best_model_transformer.ckpt")


if __name__ == '__main__':
	train()
