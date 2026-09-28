# Diffusion Model for Human Face Generation (128x128)

<img src="https://github.com/rijulpaul/ddpm-ffhq-128/blob/main/results.png?raw=true" />

  <a href="https://huggingface.co/spaces/rijulpaul/ddpm-ffhq-128" target="_blank">
    <img alt="Hugging Face Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-ffc107?color=ffc107&logoColor=white" />
  </a>
  <span>&nbsp</span>
  <a href="https://huggingface.co/rijulpaul/ddpm-ffhq-128" target="_blank">
    <img alt="Hugging Face Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" />
  </a>
  <span>&nbsp</span>
  <a href="https://github.com/rijulpaul/ddpm-ffhq-128" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Github-Repo-black?logo=github" />
  </a>
  <span>&nbsp</span>
  <a href="https://www.kaggle.com/code/rijulpaul/diffusion-model-on-ffhq" target="_blank">
    <img alt="Kaggle Notebook" src="https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle" />
  </a>
  <span>&nbsp</span>
  <a href="https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq" target="_blank">
    <img alt="Kaggle Dataset" src="https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle" />
  </a>

## Contents

* `src/` – training code
* `notebook/` – training notebook

## Overview

This project uses a DDPM based Diffusion techinique to learn to generate human faces from noise. The dataset used is ![flickr-faces-ffhq](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)

## Setup

```bash
git clone https://github.com/rijulpaul/ddpm-ffhq-128
cd ddpm-ffhq-128
```
#### Create and activate a virtual environment (recommended)
**Linux/MacOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### Install dependencies
```python
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
```bash
echo "HF_TOKEN=your_hf_token" >> .env # to upload checkpoint to hugging face hub
python train.py
```

### Run Notebook
#### On Kaggle
Add the [arnaud58/flickrfaceshq-dataset-ffhq](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) dataset as input and run.
#### On local setup
Download the [arnaud58/flickrfaceshq-dataset-ffhq](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) dataset and change the path to dataset in notebook.
