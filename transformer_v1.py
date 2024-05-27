import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW, get_scheduler
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from lagllama.lag_llama.gluon.estimator import LagLlamaEstimator
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from itertools import islice
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging
import datetime
import os
from model_v1 import diagnose_data_issues

def prepare_data_for_transformer(historical_data, odds_data, window_size=3):
    logging.info("Preparing data...")

    # Preprocess historical data
    historical_data['date_temp'] = pd.to_datetime(historical_data['date'], format='%m/%d/%y')
    historical_data['season'] = historical_data['date_temp'].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)
    historical_data.drop(columns=['date_temp'], inplace=True)
    historical_data = historical_data.sort_values('date')
    del historical_data['mp_opp']

    def add_target(team):
        team['target'] = team['won'].shift(-1)
        return team

    historical_data = historical_data.groupby('team', group_keys=False).apply(add_target)
    historical_data.loc[pd.isnull(historical_data['target']), 'target'] = 2

    diagnose_data_issues(historical_data, "Pre-scaling Check")

    # Preserve the important columns for later use
    categorical_columns = ['team', 'team_opp', 'winner']
    removed_columns = ['date', 'won', 'target', 'season']
    selected_columns = historical_data.columns[~historical_data.columns.isin(removed_columns + categorical_columns)]

    # Scale numerical columns
    scaler = MinMaxScaler()
    historical_data[selected_columns] = scaler.fit_transform(historical_data[selected_columns])

    diagnose_data_issues(historical_data[selected_columns], "Post-scaling Check")

    # Merge historical data with odds data
    game_data = pd.merge(historical_data, odds_data, left_on=["date", "team"], right_on=["date", "home"], how="left")
    game_data = pd.merge(game_data, odds_data, left_on=["date", "team_opp"], right_on=["date", "away"], how="left", suffixes=("_home", "_away"))
    diagnose_data_issues(game_data, "Post-Merge")
    print("Columns after merging historical data with odds data:", game_data.columns)

    game_data['game_id'] = game_data.groupby(['date', 'team']).ngroup()
    print("Columns after adding 'game_id':", game_data.columns)

    logging.info("Adding home advantage feature...")
    game_data['home_advantage'] = np.where(game_data['team'] == game_data['home_home'], 1, 0)
    print("Columns after adding home advantage feature:", game_data.columns)

    logging.info("Adding win/loss streak features...")
    print("Columns before adding team features:", game_data.columns)
    
    def add_win_loss_streak(df, num_games):
        df = df.sort_values(["date", "team"])
        df['result'] = np.where(df['pts'] > df['pts_opp'], 'W', 'L')

        def calc_streak(team_data):
            streak = 0
            streaks = []
            for row in team_data.itertuples(index=False):
                if row.result == 'W':
                    streak += 1
                else:
                    streak = 0
                streaks.append(streak)
            return streaks

        df['win_streak'] = df.groupby("team").apply(lambda x: pd.Series(calc_streak(x)), include_groups=False).reset_index(drop=True).shift(num_games)
        df['loss_streak'] = df.groupby("team").apply(lambda x: pd.Series(calc_streak(x[::-1])), include_groups=False).reset_index(drop=True).shift(num_games)

        return df.drop(columns='result')

    game_data = add_win_loss_streak(game_data, num_games=3)
    diagnose_data_issues(game_data, "Post-Feature Engineering")

    logging.info("Adding team features...")
    print("Columns before adding team features:", game_data.columns)
    game_data = add_team_features(game_data)
    diagnose_data_issues(game_data, "Post-Team Features")
    print("Columns after adding team features:", game_data.columns)

    logging.info("Adding opponent features...")
    print("Columns before adding opponent features:", game_data.columns)
    game_data = add_opponent_features(game_data)
    diagnose_data_issues(game_data, "Post-Opponent Features")
    print("Columns after adding opponent features:", game_data.columns)

    logging.info("Determining the winner of each game...")
    game_data['winner'] = np.where(game_data['pts'] > game_data['pts_opp'], game_data['team'], game_data['team_opp'])
    print("Columns after determining the winner:", game_data.columns)

    # Encode categorical features
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        game_data[column] = le.fit_transform(game_data[column])
        label_encoders[column] = le

    logging.info("Creating sequences for Transformer...")
    transformer_seqs, transformer_targets = create_sequences(game_data, window_size, 'winner')

    logging.info("Creating Transformer dataset...")
    X_transformer, y_transformer = create_transformer_dataset(transformer_seqs, transformer_targets)
    diagnose_data_issues(pd.DataFrame(X_transformer.reshape(X_transformer.shape[0], -1)), "X_transformer_train")

    logging.info("Data preparation completed.")
    return X_transformer, y_transformer, label_encoders

def add_team_features(df):
    logging.info("Adding team features...")
    
    def calc_team_stat(df, col, prefix):
        team_stats = df.groupby(["game_id", "team"])[col].first()
        team_stats_aligned = team_stats.reset_index().rename(columns={col: f"{prefix}_{col}"})
        df = pd.merge(df, team_stats_aligned, on=["game_id", "team"], how="left")
        return df

    for col in ["fg", "fga", "fg%", "3p", "3pa", "3p%", "ft", "fta", "ft%", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"]:
        df = calc_team_stat(df, col, "team")
        df = calc_team_stat(df, col, "opp")
        df[f"{col}_diff"] = df[f"team_{col}"] - df[f"opp_{col}"]
    
    logging.info("Adding team historical statistics features...")
    def calc_team_stat_avg(df, col, prefix):
        team_stats_avg = df.groupby("team")[col].mean()
        team_stats_avg_aligned = team_stats_avg.reset_index().rename(columns={col: f"{prefix}_{col}_avg"})
        df = pd.merge(df, team_stats_avg_aligned, on="team", how="left")
        return df

    for col in ["fg", "fga", "fg%", "3p", "3pa", "3p%", "ft", "fta", "ft%", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"]:
        df = calc_team_stat_avg(df, col, "team")
        df = calc_team_stat_avg(df, col, "opp")
    
    return df

def add_opponent_features(df):
    logging.info("Adding opponent features...")
    
    def calc_opp_stat_diff(df, col):
        def get_opp_stat(group):
            opp_group = group[group['team'] != group['team']]
            if not opp_group.empty:
                return opp_group.iloc[0][col]
            else:
                return 0  # Return 0 or any other appropriate default value when no opponent is found

        opp_stats = df.groupby("game_id").apply(get_opp_stat, include_groups=False)
        opp_stats_aligned = opp_stats.reindex(df.index)
        df[f'opp_{col}_diff'] = df[col] - opp_stats_aligned.values
        return df
    
    for col in ["fg", "fga", "fg%", "3p", "3pa", "3p%", "ft", "fta",
    "ft%", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"]:
        df = calc_opp_stat_diff(df, col)
    
    logging.info("Adding opponent historical statistics features...")
    def calc_opp_stat_avg(df, col):
        def get_opp_stat(group):
            opp_group = group[group["team"] != group["team"]]
            if not opp_group.empty:
                return opp_group[col].mean()
            else:
                return np.nan

        opp_stats = df.groupby("game_id").apply(get_opp_stat)
        opp_stats_aligned = opp_stats.reindex(df.index)
        df[f"opp_{col}_avg"] = opp_stats_aligned.values
        return df

    for col in ["fg", "fga", "fg%", "3p", "3pa", "3p%", "ft", "fta", "ft%", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"]:
        df = calc_opp_stat_avg(df, col)
    
    return df

def create_sequences(data, window_size, target_col):
    logging.info("Creating sequences...")
    game_ids = data['game_id'].unique()
    X_seqs = []
    y_seqs = []

    feature_columns = [
        "home_advantage", "win_streak", "loss_streak",
        "team_fg", "team_fga", "team_fg%", "team_3p", "team_3pa", "team_3p%", "team_ft", "team_fta", "team_ft%",
        "team_orb", "team_drb", "team_trb", "team_ast", "team_stl", "team_blk", "team_tov", "team_pf",
        "opp_fg", "opp_fga", "opp_fg%", "opp_3p", "opp_3pa", "opp_3p%", "opp_ft", "opp_fta", "opp_ft%",
        "opp_orb", "opp_drb", "opp_trb", "opp_ast", "opp_stl", "opp_blk", "opp_tov", "opp_pf",
        "fg_diff", "fga_diff", "fg%_diff", "3p_diff", "3pa_diff", "3p%_diff", "ft_diff", "fta_diff", "ft%_diff",
        "orb_diff", "drb_diff", "trb_diff", "ast_diff", "stl_diff", "blk_diff", "tov_diff", "pf_diff",
        "team_fg_avg", "team_fga_avg", "team_fg%_avg", "team_3p_avg", "team_3pa_avg", "team_3p%_avg",
        "team_ft_avg", "team_fta_avg", "team_ft%_avg", "team_orb_avg", "team_drb_avg", "team_trb_avg",
        "team_ast_avg", "team_stl_avg", "team_blk_avg", "team_tov_avg", "team_pf_avg",
        "opp_fg_avg", "opp_fga_avg", "opp_fg%_avg", "opp_3p_avg", "opp_3pa_avg", "opp_3p%_avg",
        "opp_ft_avg", "opp_fta_avg", "opp_ft%_avg", "opp_orb_avg", "opp_drb_avg", "opp_trb_avg",
        "opp_ast_avg", "opp_stl_avg", "opp_blk_avg", "opp_tov_avg", "opp_pf_avg",
        "team", "team_opp"
    ]

    for i in range(len(game_ids) - window_size):
        seq_games = game_ids[i:i + window_size]
        sequence = data[data['game_id'].isin(seq_games)]
        features = sequence[feature_columns]
        target = sequence.iloc[-1][target_col]

        X_seqs.append(features.values)
        y_seqs.append(target)

    logging.info("Sequences created.")
    return X_seqs, y_seqs

def create_transformer_dataset(sequences, targets):
    logging.info("Creating Transformer dataset...")
    X = np.array([seq for seq in sequences])
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit and transform the targets
    y = label_encoder.fit_transform(targets)
    
    logging.info("Transformer dataset created.")
    return X, y

class NBADataset(Dataset):
    def __init__(self, data, labels, context_length):
        self.data = data
        self.labels = labels
        self.context_length = context_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx][:self.context_length]
        if len(input_ids) < self.context_length:
            input_ids = np.pad(input_ids, ((0, self.context_length - len(input_ids)), (0, 0)), 'constant')
        label = self.labels[idx]
        return torch.tensor(input_ids, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_lag_llama_predictions(dataset, prediction_length, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        nonnegative_pred_samples=nonnegative_pred_samples,
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },
        batch_size=batch_size,
        num_parallel_samples=num_samples,
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

def run_pipeline(historical_data, odds_data):
    logging.info("Running pipeline...")

    X_transformer, y_transformer, label_encoders = prepare_data_for_transformer(historical_data, odds_data)

    context_length = 64  # Adjust as necessary
    train_size = int(0.8 * len(X_transformer))
    val_size = len(X_transformer) - train_size
    train_dataset = NBADataset(X_transformer[:train_size], y_transformer[:train_size], context_length)
    val_dataset = NBADataset(X_transformer[train_size:], y_transformer[train_size:], context_length)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load Lag-Llama model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=1,  # Set according to your task
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        nonnegative_pred_samples=True,
        batch_size=64,
        num_parallel_samples=20,
        trainer_kwargs={"max_epochs": 50},  # Set according to your task
    )

    # Fine-tune Lag-Llama model
    predictor = estimator.train(train_dataloader, cache_data=True, shuffle_buffer_length=1000)

    # Save the fine-tuned model
    model_path = "fine_tuned_lag_llama.ckpt"
    torch.save(predictor.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Make predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=val_dataloader,
        predictor=predictor,
        num_samples=20
    )

    forecasts = list(tqdm(forecast_it, total=len(val_dataloader), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(val_dataloader), desc="Ground truth"))

    # Plot results
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})

    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9
    ):
        ax = plt.subplot(3, 3, idx + 1)
        plt.plot(ts[-4 * context_length:].to_timestamp(), label="target")
        forecast.plot(color='g')
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()

    # Evaluate model
    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
    print(agg_metrics)

    logging.info("Pipeline completed.")
    return predictor

# Example usage
odds_data = pd.read_csv("/mnt/data/combined_od_v1.csv")
historical_data = pd.read_csv("/mnt/data/2012_to_2024_data.csv")

logging.info("Starting pipeline...")
predictor = run_pipeline(historical_data, odds_data)