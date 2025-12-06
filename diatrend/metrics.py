import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def calculate_rmse(preds, targets):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return np.sqrt(np.mean((preds - targets) ** 2))

def clarke_error_grid(preds, targets):
    """
    assigns zones (a, b, c, d, e) to prediction-target pairs.
    inputs can be torch tensors or numpy arrays.
    """
    if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy().flatten()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy().flatten()

    zones = []
    for p, t in zip(preds, targets):
        if t == p:
            zones.append('A')
        elif (t >= 70 and t <= 180 and p >= 70 and p <= 180):
            zones.append('A')
        elif abs(p - t) <= 20:
            zones.append('A')
        elif (t < 70 and p < 70) or (t > 180 and p > 180):
            zones.append('B')
        elif (t < 70 and p > 180) or (t > 180 and p < 70):
            zones.append('E')
        elif (t >= 70 and t <= 180 and (p < 70 or p > 180)) or (p >= 70 and p <= 180 and (t < 70 or t > 180)):
            zones.append('C')
        else:
            zones.append('D')
    return zones

def plot_clarke_grid(preds, targets, save_path=None):
    zones = clarke_error_grid(preds, targets)
    zone_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'E': 'purple'}
    
    if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, preds, c=[zone_colors[z] for z in zones], s=10, alpha=0.6)
    ax.plot([0, 400], [0, 400], color='black', linestyle='--', linewidth=1)
    ax.set_title("clarke error grid")
    ax.set_xlabel("reference (mg/dl)")
    ax.set_ylabel("predicted (mg/dl)")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig, Counter(zones)