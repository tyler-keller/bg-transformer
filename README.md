## How the code is structured:
The `data` folder holds the original and cleaned dataset. 
The original dataset is a set of 54 .xlxs files. 
The cleaned dataset is a single subject's cleaned data that can be trained on by the model.

> [!NOTE]  
> If you want to run the code, you will have to unzip the DiaTrend dataset from [DiaTrend@Synapse](https://www.synapse.org/Synapse:syn38187184/wiki/619490).
> Additionally, you'll have to exectue the `clean_data.py` script for your intended subject file(s).

The `models` folder holds different trained models. 
The file naming structure indicates the model architecture and model training specifics.

The `src` folder holds the model.py, train.py and clean_data.py scripts. 
model.py is the PyTorch model object. 
train.py is a one-off script w/ argument parsing for CLI training. 
clean_data.py is a one-off script w/ argument parsing for CLI data cleaning.

## How to run the code:

> [!IMPORTANT]  
> Assuming development environment in Linux, MacOS, or WSL.

#### Virtualenv

> [!NOTE]  
> In the `diatrend` root directory.

> [!IMPORTANT]  
> Assuming a Python 3.12.8 version. Package conflicts may result if this is not the case.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

TODO: create Dockerfile for containerization

#### Data Cleaning/Preprocessing

> [!NOTE]  
> In the `diatrend/src` directory

```bash
python clean_data.py --verbose --subject_id 4 --to_csv
```


#### Model Training/Testing

> [!NOTE]  
> In the `diatrend/src` directory

```bash
python train.py --epochs 10 --batch_size 32 --learning_rate 0.0001 --emb_dim 64 --num_heads 8 --ff_dim 128 --verbose --output_dir "../output"
```

## Contributors:
- [Nathan Panzer](https://github.com/TheNathanSpace)
- [Daniel Arinze](https://github.com/Daniel-Arinze-hub)