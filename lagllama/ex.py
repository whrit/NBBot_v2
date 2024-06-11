import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import logging
import talib
import yfinance as yf
import tqdm
from itertools import islice
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from lag_llama.gluon.estimator import LagLlamaEstimator
from transformers import AdamW, get_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
print(f"Using device: {device}")  # Optional: to confirm the device being used

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Fetch SPY data
logging.info("Fetching stock data...")
stock_data = yf.download('SPY', start='2004-01-01', end='2024-05-17')

# Swap "Adj Close" data into the "Close" column
logging.info("Swapping 'Adj Close' data into 'Close' column...")
stock_data['Close'] = stock_data['Adj Close']

# Remove the "Adj Close" column
logging.info("Removing 'Adj Close' column...")
stock_data = stock_data.drop(columns=['Adj Close'])

# Checking for missing values
logging.info("Checking for missing values...")
logging.info(stock_data.isnull().sum())

# Filling missing values, if any
logging.info("Filling missing values, if any...")
stock_data.ffill(inplace=True)
stock_data.dropna(inplace=True)

logging.info("Calculating technical indicators...")
stock_data['SMA_10'] = talib.SMA(stock_data['Close'], timeperiod=10)
stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
stock_data['EMA_20'] = talib.EMA(stock_data['Close'], timeperiod=20)
stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
stock_data['STOCH_K'], stock_data['STOCH_D'] = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
stock_data['MACD'], stock_data['MACDSIGNAL'], stock_data['MACDHIST'] = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock_data['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])
stock_data['ATR'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['BBANDS_UPPER'], stock_data['BBANDS_MIDDLE'], stock_data['BBANDS_LOWER'] = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
stock_data['MOM'] = talib.MOM(stock_data['Close'], timeperiod=10)
stock_data['CCI'] = talib.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['WILLR'] = talib.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['TSF'] = talib.TSF(stock_data['Close'], timeperiod=14)
stock_data['TRIX'] = talib.TRIX(stock_data['Close'], timeperiod=30)
stock_data['ULTOSC'] = talib.ULTOSC(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
stock_data['ROC'] = talib.ROC(stock_data['Close'], timeperiod=10)
stock_data['PLUS_DI'] = talib.PLUS_DI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['MINUS_DI'] = talib.MINUS_DI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['PLUS_DM'] = talib.PLUS_DM(stock_data['High'], stock_data['Low'], timeperiod=14)
stock_data['MINUS_DM'] = talib.MINUS_DM(stock_data['High'], stock_data['Low'], timeperiod=14)

logging.info("Checking for NaN values after calculating technical indicators...")
logging.info(stock_data.isnull().sum())

stock_data.index = pd.to_datetime(stock_data.index)

# Split the data into training, validation, and testing sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_size = int(len(stock_data) * train_ratio)
val_size = int(len(stock_data) * val_ratio)
test_size = len(stock_data) - train_size - val_size

train_data = stock_data[:train_size]
val_data = stock_data[train_size:train_size+val_size]
test_data = stock_data[train_size+val_size:]

# Create validation dataset
start_date_val = pd.Timestamp(val_data.index[0])
start_date_train = pd.Timestamp(train_data.index[0])
start_date_test = pd.Timestamp(test_data.index[0])

# Select target column and features
target_column = 'Close'  
feature_columns = ['Open', 'High', 'Low', 'Volume']  

val_dataset = ListDataset(
    [{"start": start_date_val, "target": val_data[target_column].values}],
    freq="B"
)

train_dataset = ListDataset(
    [{"start": start_date_train, "target": train_data[target_column].values}],
    freq="B"  # Assuming your stock data is Business Day frequency
)

test_dataset = ListDataset(
    [{"start": start_date_train, "target": train_data[target_column].values}],
    freq="B"  # Assuming your stock data is Business Day frequency
)

from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="lag_llama_{epoch:02d}_{train_loss:.4f}",
    save_top_k=1,
    monitor="train_loss",
    mode="min",
)

from lightning.pytorch.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
)

# 4. Update get_lag_llama_predictions()
def get_lag_llama_predictions(dataset, prediction_length, context_length=32, num_samples=20, batch_size=64, device="cuda"):
    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    print(estimator_args)  # Print configurations to ensure they are used

    torch.set_float32_matmul_precision('high')
    # Create estimator using configurations from checkpoint
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        num_parallel_samples=num_samples,
        batch_size=batch_size,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss

# 3. Fine-tune Lag-Llama (with Hyperparameter Tuning)
prediction_length = 24  
context_length = prediction_length * 3
num_samples = 20

best_agg_metrics = {}
best_predictor = None  # Initialize to None


# Load the checkpoint (Same as in `get_lag_llama_predictions`)
ckpt = torch.load("lag-llama.ckpt", map_location=device)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
print(estimator_args)  # Print configurations to ensure they are used

torch.set_float32_matmul_precision('high')
# Create estimator using configurations from checkpoint
estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,

        # distr_output="neg_bin",
        # scaling="mean",
        nonnegative_pred_samples=True,
        aug_prob=0,
        lr=5e-4,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        time_feat=estimator_args["time_feat"],

        # rope_scaling={
        #     "type": "linear",
        #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        # },

        batch_size=64,
        num_parallel_samples=num_samples,
        trainer_kwargs={
            "max_epochs": 50,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "callbacks": [checkpoint_callback, early_stop_callback],
        },
    )

# Train the estimator
predictor = estimator.train(train_dataset, val_dataset=val_dataset, cache_data=True, shuffle_buffer_length=1000)

# Generate predictions on the test dataset (not the training dataset)
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,  # Use test_dataset for evaluation
    predictor=predictor,
    num_samples=num_samples
)

# Convert iterators to lists
forecasts = list(tqdm.tqdm(forecast_it, total=len(test_dataset), desc="Forecasting batches"))
tss = list(tqdm.tqdm(ts_it, total=len(test_dataset), desc="Ground truth"))

# Evaluate and store the metrics for this configuration
evaluator = Evaluator()
agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
print(f'Aggregate metrics: {agg_metrics}')

best_estimator = estimator  # Save the estimator object
print("New best model found!")

# Identify the best model based on the chosen metric (e.g., MASE)
# 5. Saving the Best Model
best_predictor_module = best_estimator.create_lightning_module()
torch.save(best_predictor_module.state_dict(), "best_lag_llama_predictor.pth") 

# Generate predictions and ground truths using the best predictor
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,  
    predictor=best_predictor,
    num_samples=num_samples
)

forecasts = list(tqdm.tqdm(forecast_it, total=len(test_dataset), desc="Forecasting batches"))
tss = list(tqdm.tqdm(ts_it, total=len(test_dataset), desc="Ground truth"))

plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx + 1)
    plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target")
    forecast.plot(color='g')
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(forecast.item_id)

plt.gcf().tight_layout()
plt.legend()
plt.show()