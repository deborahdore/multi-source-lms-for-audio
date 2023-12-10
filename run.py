from __future__ import print_function

import lightning as L
import torchaudio
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

import config
from data import SlakhDataset
from model import VQVAE


def generate_sound(model, dataloader):
	instruments_name = ["bass.wav", "drums.wav", "guitar.wav", "piano.wav"]
	mixture, instruments = dataloader
	output = model(dataloader).squeeze()
	instruments = instruments.squeeze()

	for idx in range(instruments.size(0)):
		torchaudio.save(uri=f'{config.OUTPUT_DIR}/real_{instruments_name[idx]}',
						src=instruments[idx].unsqueeze(0),
						sample_rate=22050)

		torchaudio.save(uri=f'{config.OUTPUT_DIR}/fake_{instruments_name[idx]}',
						src=output[idx].unsqueeze(0),
						sample_rate=22050)


if __name__ == '__main__':
	train = DataLoader(SlakhDataset(config.TRAIN_DIR), batch_size=config.BATCH_SIZE, shuffle=True)
	test = DataLoader(SlakhDataset(config.TEST_DIR), batch_size=config.BATCH_SIZE, shuffle=False)
	val = DataLoader(SlakhDataset(config.VAL_DIR), batch_size=config.BATCH_SIZE, shuffle=False)

	vq_vae = VQVAE(num_hidden=config.NUM_HIDDEN,
				   num_residual_layer=config.NUM_RESIDUAL_LAYER,
				   num_residual_hidden=config.NUM_RESIDUAL_HIDDEN,
				   num_embedding=config.NUM_EMBEDDINGS,
				   embedding_dim=config.EMBEDDING_DIM,
				   commitment_cost=config.COMMITMENT_COST,
				   learning_rate=config.LEARNING_RATE)

	if config.WANDB:
		logger = WandbLogger(project=config.WANDB_PROJECT_NAME,
							 save_dir=config.LOGGING_DIR,
							 offline=config.WANDB_OFFLINE)
	else:
		logger = TensorBoardLogger(save_dir=config.LOGGING_DIR)

	trainer = L.Trainer(limit_train_batches=config.LIMIT_TRAIN_BATCHES,
						max_epochs=config.MAX_EPOCHS,
						callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
						profiler=config.PROFILER,
						logger=logger)

	trainer.fit(model=vq_vae, train_dataloaders=train, val_dataloaders=val)
	vq_vae.eval()
	trainer.test(model=vq_vae, dataloaders=test)

	generate_sound(vq_vae, next(iter(DataLoader(test.dataset, batch_size=1))))
