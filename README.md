# Fashion MNIST ANN Pipeline

Fully-connected ANN that classifies Fashion-MNIST images into 10 clothing
categories. Data and model files are versioned with DVC using Google Drive
as the remote. The whole pipeline runs with a single `dvc repro`.

## Setup

    python -m venv venv
    source venv/bin/activate
    pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib seaborn pandas

## Run

    dvc repro      # runs prepare -> preprocess -> train -> evaluate
    dvc push       # uploads data and model to the Google Drive remote

## Layout

    src/prepare.py      downloads Fashion-MNIST into data/raw/
    src/preprocess.py   normalizes and splits, writes data/processed/
    src/train.py        trains the ANN, writes models/model.h5
    src/evaluate.py     writes metrics.json and the confusion matrix
    params.yaml         all hyperparameters
    dvc.yaml            pipeline stage definitions
