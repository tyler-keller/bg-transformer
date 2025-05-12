import argparse
import logging
import math
import os
import time
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GlucoseDataset
from model import GlucoseTransformerEncoder

parser = argparse.ArgumentParser(prog = 'train model')
parser.add_argument('--data_dir', default = '../data/cleaned/slow_iob/5', type = str, help = 'specify data directory')
parser.add_argument('--seq_len', default = 6, type = int, help = 'specify training sequence length')
parser.add_argument('--emb_dim', default = 64, type = int, help = 'embedding dimension')
parser.add_argument('--num_heads', default = 8, type = int, help = 'number of attention heads')
parser.add_argument('--ff_dim', default = 128, type = int, help = 'feedforward layer dimensions')
parser.add_argument('--epochs', required = True, type = int, help = 'number of epochs')
parser.add_argument('--batch_size', required = True, type = int, help = 'batch size')
parser.add_argument('--learning_rate', required = True, type = float, help = 'learning rate')
parser.add_argument('--verbose', action = 'store_true', help = 'print debug messages?')
parser.add_argument('--output_dir', type = str, help = 'specify output directory')
args = parser.parse_args()

# turning all the above into their own local vars
data_dir = args.data_dir
seq_len = args.seq_len
emb_dim = args.emb_dim
num_heads = args.num_heads
ff_dim = args.ff_dim
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
verbose = args.verbose
if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = Path(os.getcwd())

instance_time: int = math.floor(time.time())
output_dir = output_dir / str(instance_time)
output_dir.mkdir(parents = True, exist_ok = True)

logger: Logger

def setup_logger():
    global logger
    logger = logging.getLogger("BG Prediction")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(output_dir / f"bg_prediction_{instance_time}.log")
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_dataset_splits(data_dir, seq_len, split = (0.7, 0.15, 0.15)):
    logger.debug("Loading dataset")

    assert sum(split) == 1.0, "splits must sum to 1.0"

    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")])

    # CORRECT SEQUENCE LOGIC -- PULLS ALL SEQ_LEN SEQUENCES PER FILE
    all_sequences = []
    for f in all_files:
        data = pd.read_csv(f).to_numpy(dtype = np.float32)
        if len(data) >= seq_len:
            for i in range(len(data) - seq_len):
                window = data[i:i + seq_len + 1]  # plus one for the correct y target
                all_sequences.append(window)

    n = len(all_sequences)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train_targets = [window[-1, 0] for window in all_sequences[:n_train]]
    mgdl_mu = np.mean(train_targets)
    mgdl_std = np.std(train_targets)

    logger.debug(f"Splitting dataset: train={n_train}, val={n_val}, test={n - n_train - n_val}")
    train_set = GlucoseDataset(all_sequences[:n_train], seq_len, mgdl_mu, mgdl_std)
    val_set = GlucoseDataset(all_sequences[n_train:n_train + n_val], seq_len, mgdl_mu, mgdl_std)
    test_set = GlucoseDataset(all_sequences[n_train + n_val:], seq_len, mgdl_mu, mgdl_std)

    logger.info(f'n_train: {n_train} | n_val: {n_val} | n_test: {n - (n_train + n_val)}')

    return train_set, val_set, test_set, mgdl_mu, mgdl_std


def clarke_error_grid(preds, targets, save_plot = False):
    logger.debug("Performing Clarke Error Grid analysis")

    zone_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'E': 'purple'}
    preds = preds.flatten().numpy()
    targets = targets.flatten().numpy()

    logger.debug("Assigning zones")
    zones = []
    for p, t in zip(preds, targets):
        if t == p:
            zones.append('A')
        elif (t >= 70 and t <= 180 and p >= 70 and p <= 180):
            zones.append('A')
        elif (abs(p - t) <= 20):
            zones.append('A')
        elif (t < 70 and p < 70) or (t > 180 and p > 180):
            zones.append('B')
        elif ((t < 70 and p > 180) or (t > 180 and p < 70)):
            zones.append('E')
        elif ((t >= 70 and t <= 180 and (p < 70 or p > 180)) or (p >= 70 and p <= 180 and (t < 70 or t > 180))):
            zones.append('C')
        else:
            zones.append('D')

    logger.debug("Saving plot")
    if save_plot:
        fig, ax = plt.subplots()
        ax.scatter(targets, preds, c = [zone_colors[z] for z in zones], s = 10)
        ax.set_title("Clarke Error Grid")
        ax.set_xlabel("Reference (mg/dL)")
        ax.set_ylabel("Predicted (mg/dL)")
        ax.plot([0, 400], [0, 400], color = 'black', linestyle = '--', linewidth = 1)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        plt.grid(True)
        plt.savefig(output_dir / f"clarke_error_grid_{instance_time}.png")
        plt.close(fig)

    return zones


def train():
    logger.debug("Starting training")
    train_set, val_set, test_set, mgdl_mu, mgdl_std = load_dataset_splits(data_dir, seq_len)

    input_dim = train_set[0][0].shape[-1]

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 2)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    logger.debug("Instantiating model")
    model = GlucoseTransformerEncoder(input_dim = input_dim, seq_len = seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss()

    logger.debug("Starting training epochs")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        logger.info(
            f"Epoch {epoch + 1}: Train Loss = {total_loss / len(train_loader):.4f} | Val Loss = {val_loss / len(val_loader):.4f}")

    # test RMSE
    logger.debug("Evaluating model")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    preds = preds * mgdl_std + mgdl_mu
    targets = targets * mgdl_std + mgdl_mu

    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
    logger.info(f"Test RMSE: {rmse.item():.4f}")

    # baseline RMSE: predict mean of training targets
    train_targets = torch.cat([y for _, y in DataLoader(train_set)], dim = 0)
    baseline_pred = train_targets.mean()
    baseline_rmse = torch.sqrt(torch.mean((baseline_pred - targets) ** 2))
    logger.info(f"Baseline RMSE (mean mg/dl): {baseline_rmse.item():.4f}")

    # sample preds vs actual
    logger.info("Sample prediction vs target:")
    for i in range(min(5, len(preds))):
        logger.info(f"pred: {preds[i].item():.2f} | actual: {targets[i].item():.2f}")

    zones = clarke_error_grid(preds, targets, save_plot = True)

    from collections import Counter
    zone_counts = Counter(zones)
    total = len(zones)

    logger.info("Clarke Error Grid Zones:")
    for z in 'ABCDE':
        count = zone_counts.get(z, 0)
        logger.info(f"Zone {z}: {count} ({(count / total) * 100:.1f}%)")

    model_path: str = f"../models/bg_transformer_epochs_{epochs}_batch_size_{batch_size}_lr_{learning_rate}_seqlen_{model.seq_len}_embdim_{model.embedding_dim}_numheads_{model.num_heads}_ffdim_{model.ff_dim}_testrmse_{rmse.item():.0f}.pth"
    logger.debug(f"Saving model to file {model_path}")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    setup_logger()
    train()
