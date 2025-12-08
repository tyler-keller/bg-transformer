import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import sys

# ensure we can find the package
sys.path.append(os.getcwd())

from diatrend.model import GlucoseTransformer
from diatrend.metrics import clarke_error_grid

st.set_page_config(page_title="diatrend", layout="wide")
st.title("diatrend model viewer")

# setup paths
MODELS_DIR = 'models'
DATA_DIR = 'data/cleaned/slow_iob/5' # hardcoded for demo, make dynamic as needed

if not os.path.exists(MODELS_DIR):
    st.error("models dir not found")
    st.stop()

# 1. select model
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
selected_file = st.sidebar.selectbox("select checkpoint", model_files)
model_path = os.path.join(MODELS_DIR, selected_file)

# 2. load model & metadata
@st.cache_resource
def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    
    model = GlucoseTransformer(
        input_dim=config['input_dim'],
        seq_len=config['seq_len'],
        embedding_dim=config.get('emb_dim', 64)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config, scaler

try:
    model, config, scaler = load_checkpoint(model_path)
    # model = load_checkpoint(model_path)
    st.sidebar.success("model loaded")
    st.sidebar.json(config)
    st.sidebar.write("scaler stats:", scaler)
except Exception as e:
    st.error(f"failed to load model: {e}")
    st.stop()

# 3. select data
data_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
selected_data = st.selectbox("select data segment", data_files)
df = pd.read_csv(os.path.join(DATA_DIR, selected_data))

# 4. prepare inference
seq_len = config['seq_len']
data_arr = df.to_numpy(dtype=np.float32)

if len(data_arr) <= seq_len:
    st.warning("segment too short for model sequence length")
else:
    # create sliding windows for the whole file
    x_wins, y_trues = [], []
    for i in range(len(data_arr) - seq_len):
        x_wins.append(data_arr[i : i+seq_len])
        y_trues.append(data_arr[i+seq_len, 0]) # target is mg/dl at next step
        
    x_tensor = torch.tensor(np.array(x_wins))
    
    with torch.no_grad():
        preds_norm = model(x_tensor).flatten().numpy()
        
    # un-scale predictions
    preds_mgdl = preds_norm * scaler['std'] + scaler['mu']
    
    # 5. visualize
    results = pd.DataFrame({'actual': y_trues, 'predicted': preds_mgdl})
    
    st.line_chart(results)
    
    # metrics
    rmse = np.sqrt(np.mean((results['actual'] - results['predicted'])**2))
    st.metric("RMSE", f"{rmse:.2f} mg/dl")
    
    # clarke grid
    zones = clarke_error_grid(results['predicted'].values, results['actual'].values)
    st.bar_chart(pd.Series(zones).value_counts())