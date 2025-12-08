import argparse
import os
import torch
import logging
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from diatrend.model import GlucoseTransformer
from diatrend.data import load_and_window_data, GlucoseDataset
from diatrend.metrics import calculate_rmse, plot_clarke_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--seq_len', type=int, default=12) # 1 hour context
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # 1. load data
    logger.info("loading data...")
    x_all, y_all = load_and_window_data(args.data_dir, args.seq_len)
    
    # split 70/15/15
    n = len(x_all)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    # 2. create datasets
    # we calculate mu/std only on train set to prevent leakage
    train_ds = GlucoseDataset(x_all[:n_train], y_all[:n_train])
    val_ds = GlucoseDataset(x_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val], mu=train_ds.mu, std=train_ds.std)
    test_ds = GlucoseDataset(x_all[n_train+n_val:], y_all[n_train+n_val:], mu=train_ds.mu, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 3. init model
    input_dim = x_all.shape[-1]
    model = GlucoseTransformer(input_dim=input_dim, seq_len=args.seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    # 4. training loop
    logger.info(f"starting training on {device}")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"epoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        logger.info(f"epoch {epoch+1} | train_loss: {avg_train:.4f} | val_loss: {avg_val:.4f}")

    # 5. testing & saving
    logger.info("evaluating on test set...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            # un-scale predictions back to mg/dl for readable metrics
            p_mgdl = p * train_ds.std + train_ds.mu
            y_mgdl = y * train_ds.std + train_ds.mu
            preds.append(p_mgdl)
            targets.append(y_mgdl)
            
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    rmse = calculate_rmse(preds, targets)
    logger.info(f"test rmse: {rmse:.2f} mg/dl")
    
    uuid_str = str(uuid.uuid4())

    # plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f'clarke_grid_{uuid_str}.png')
    plot_clarke_grid(preds, targets, save_path=plot_path)

    # save comprehensive checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'id': uuid_str,
        'config': {
            'input_dim': input_dim,
            'seq_len': args.seq_len,
            'emb_dim': model.embedding_dim,
            'num_heads': model.num_heads,
            'ff_dim': model.ff_dim,
            'dropout': model.dropout,
        },
        'scaler': {
            'mu': float(train_ds.mu),
            'std': float(train_ds.std)
        },
        'metrics': {'rmse': rmse}
    }
    
    save_path = os.path.join(args.output_dir, f'bg_transformer_epochs_{args.epochs}_batch_{args.batch_size}_lr_{args.lr}_seq_{args.seq_len}_emb_{model.embedding_dim}_numheads_{model.num_heads}_ffdim_{model.ff_dim}_testrmse_{rmse:.0f}.pth')
    torch.save(checkpoint, save_path)
    logger.info(f"saved model to {save_path}")

if __name__ == "__main__":
    train()