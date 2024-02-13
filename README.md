<div align="center">

# Multi Source Large Language Model For Audio

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repo explores the potential of leveraging a large language model (LLM), such as BERT, trained on the vector
quantized representation of audio to enhance music generation and music separation tasks. Specifically, it provides the
code to: (1) train a Vector Quantized Variational Autoencoder (VQVAE) on the Synthesized Lakh Dataset (Slakh), (2) train
a simple Transformer architecture on the vector quantized representation of the audio, and (3) fine-tune a basic BERT
model on the indexes of the embeddings of the codebook of the audio.

Tasks (1) and (2) employ two different strategies on purpose, given that basic BERT model can't handle sequences longer
than 512 tokens directly. Its input usually involves tokenized text, converted into numerical representations using an
embedding layer. These representations are usually integers, representing indices in the model's vocabulary. Therefore,
in this case, it was necessary to represent audio data in a format compatible with BERT, such as the indexes of the
codebook more similar to the input audio that would have been used to quantize the input source.

## Installation

#### Conda

```bash
# clone project
git clone https://github.com/deborahdore/multi-source-lms-for-audio
cd multi-source-lms-for-audio

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

### Basic Training with default configurations

```bash
# train on CPU
python src/train.py trainer.accelerator=cpu

# train on GPU
python src/train.py trainer.accelerator=gpu
```

#### Choose the model to train with flag: **[train_vqvae, train_transformer, train_bert]**

```bash
python src/train.py debug=default train_bert=True
```

Modify configuration directly in the [configuration](configs) folder.

### Debug

```bash
python src/train.py debug=default
```

Modify configuration for debugging in the [default.yaml](configs%2Fdebug%2Fdefault.yaml).

### Hyper-parameter search

```bash
python src/train.py hparams_search=optuna
```

Modify hyper-parameter search configurations in the [optuna.yaml](configs%2Fhparams_search%2Foptuna.yaml)

## Dataset

The dataset used in this project is the Slakh2100 dataset, which is available at the
following [URL](https://zenodo.org/records/4603870). <br>
The dataset was processed to extract the four most prevalent instruments: bass, drums, guitar, and piano. <br>
The final dataset should have this structure

```
    slakh2100
        |
        | - train
        |     | 
        |     | - Track01
        |     |     | - bass.wav
        |     |     | - guitar.wav
        |     |     | - drums.wav
        |     |     | - piano.wav
        |     |
        |     | - Track02
        |     | - ...
        | - validation
        |     | 
        |     | - Track03
        |     | - ...
        | - test
              | 
              | - Track04
              | - ...
```

A simple guide on how to do that can be
found [here](https://github.com/gladia-research-group/multi-source-diffusion-models/tree/main/data).

## Model Checkpoints

Model checkpoints are available in their corresponding folder: [best_checkpoint](logs%2Fbest_checkpoint)