import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, Dropout, Multiply, Bidirectional, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import tensorrt
import os
import logging
import datetime
import pickle

def cpugpu():
    # Check if GPU is available and print the list of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found GPU: {gpu}")
    else:
        print("No GPU devices found.")
    
    if gpus:
        try:
            # Specify the GPU device to use (e.g., use the first GPU)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Test TensorFlow with a simple computation on the GPU
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = a * b

            print("GPU is available and TensorFlow is using it.")
            print("Result of the computation on GPU:", c.numpy())
        except RuntimeError as e:
            print("Error while setting up GPU:", e)
    else:
        print("No GPU devices found, TensorFlow will use CPU.")
        
logging.basicConfig(filename='logfile.log', filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def diagnose_data_issues(data, label):
    print(f"--- Diagnosing Data Issues in {label} ---")
    # Convert to a DataFrame if not already one
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            # Reshape 3-dimensional array to 2-dimensional
            data = data.reshape(data.shape[0], -1)
        data = pd.DataFrame(data)
    # Handle possible cases where data is not numeric
    numeric_df = data.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric data to diagnose.")
        return
    
    # Check for NaN and Inf values
    contains_nan = numeric_df.isna().any().any()
    contains_inf = np.isinf(numeric_df.to_numpy()).any()
    max_value = numeric_df.max().max()
    min_value = numeric_df.min().min()

    print("Contains NaN: ", contains_nan)
    print("Contains Inf: ", contains_inf)
    print("Max value: ", max_value)
    print("Min value: ", min_value)

    # Save DataFrame with diagnostics to CSV for inspection
    numeric_df.to_csv(f'{label}_diagnostics.csv', index=False)
    print(f"Saved {label} diagnostics to {label}_diagnostics.csv")

def prepare_data(historical_data, odds_data, window_size=3):
    logging.info("Preparing data...")

    # Preprocess historical data
    historical_data['date_temp'] = pd.to_datetime(historical_data['date'], format='%m/%d/%y')
    historical_data['season'] = historical_data['date_temp'].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)
    historical_data.drop(columns=['date_temp'], inplace=True)
    historical_data = historical_data.sort_values('date')
    # del historical_data['mp']
    del historical_data['mp_opp']

    def add_target(team):
        team['target'] = team['won'].shift(-1)
        return team

    historical_data = historical_data.groupby('team', group_keys=False).apply(add_target)
    historical_data.loc[pd.isnull(historical_data['target']), 'target'] = 2

    diagnose_data_issues(historical_data, "Pre-scaling Check")

    removed_columns = ['team', 'date', 'won', 'target', 'team_opp', 'season']
    selected_columns = historical_data.columns[~historical_data.columns.isin(removed_columns)]

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

    logging.info("Creating sequences for LSTM...")
    lstm_seqs, lstm_targets = create_sequences(game_data, window_size, 'winner')

    for i, seq in enumerate(lstm_seqs):
        print(f"Data types in sequence {i}:")
        print(pd.DataFrame(seq).dtypes)

    logging.info("Creating LSTM dataset...")
    X_lstm, y_lstm = create_lstm_dataset(lstm_seqs, lstm_targets)
    diagnose_data_issues(pd.DataFrame(X_lstm.reshape(X_lstm.shape[0], -1)), "X_lstm_train")

    logging.info("Creating XGBoost dataset...")
    X_xgb, y_xgb, groups_xgb = create_xgb_dataset(game_data)
    diagnose_data_issues(X_xgb, "X_xgb_train")

    # Remove non-numeric columns from X_xgb
    numeric_columns = X_xgb.select_dtypes(include=[np.number]).columns
    X_xgb = X_xgb[numeric_columns]

    logging.info("Splitting data into train and test sets...")
    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
    X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
    groups_xgb_train, groups_xgb_test = train_test_split(groups_xgb, test_size=0.2, random_state=42)

    diagnose_data_issues(X_lstm_train, "X_lstm_train")
    diagnose_data_issues(X_xgb_train, "X_xgb_train")

    # Handle infinite values (if necessary)
    X_lstm_train[np.isinf(X_lstm_train)] = 0
    X_lstm_test[np.isinf(X_lstm_test)] = 0

    # Convert X_lstm_train and X_lstm_test to DataFrames
    X_lstm_train_df = pd.DataFrame(X_lstm_train.reshape(X_lstm_train.shape[0], -1))
    X_lstm_test_df = pd.DataFrame(X_lstm_test.reshape(X_lstm_test.shape[0], -1))

    # Align the data by finding common indices
    common_indices = X_xgb_train.index.intersection(X_lstm_train_df.index)
    X_xgb_train = X_xgb_train.loc[common_indices]
    y_xgb_train = y_xgb_train.loc[common_indices]
    X_lstm_train = X_lstm_train[common_indices]
    y_lstm_train = y_lstm_train[common_indices]

    # Align the test data
    common_indices_test = X_xgb_test.index.intersection(X_lstm_test_df.index)
    X_xgb_test = X_xgb_test.loc[common_indices_test]
    y_xgb_test = y_xgb_test.loc[common_indices_test]
    X_lstm_test = X_lstm_test[common_indices_test]
    y_lstm_test = y_lstm_test[common_indices_test]

    # Reshape X_lstm_train
    X_lstm_train = X_lstm_train.reshape(X_lstm_train.shape[0], window_size, -1)

    # Check data types of X_lstm_train and X_lstm_test
    print("Data types in X_lstm_train:")
    print(pd.DataFrame(X_lstm_train[0]).dtypes)
    print("Data types in X_lstm_test:")
    print(pd.DataFrame(X_lstm_test[0]).dtypes)

    numeric_columns = pd.DataFrame(X_lstm_train[0]).select_dtypes(include=[np.number]).columns
    X_lstm_train = X_lstm_train[:, :, numeric_columns]
    X_lstm_test = X_lstm_test[:, :, numeric_columns]

    # Reshape X_lstm_test
    X_lstm_test = X_lstm_test.reshape(X_lstm_test.shape[0], window_size, -1)

    print("Shape of X_lstm_train:", X_lstm_train.shape)
    print("Shape of X_lstm_test:", X_lstm_test.shape)

    X_lstm_train = X_lstm_train.astype(np.float32)
    X_lstm_test = X_lstm_test.astype(np.float32)
    y_lstm_train = y_lstm_train.astype(np.float32)
    y_lstm_test = y_lstm_test.astype(np.float32)

    from keras.utils import to_categorical

    # Convert the target to one-hot encoded format
    y_lstm_train = to_categorical(y_lstm_train)
    y_lstm_test = to_categorical(y_lstm_test)

    # Encode the target variable for XGBoost
    label_encoder = LabelEncoder()
    y_xgb_train = label_encoder.fit_transform(y_xgb_train)
    y_xgb_test = label_encoder.transform(y_xgb_test)

    logging.info("Data preparation completed.")
    return (X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test), (X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test, groups_xgb_train, groups_xgb_test), selected_columns

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

def add_betting_features(df, betting_data):
    logging.info("Adding betting features...")
    df["betting_line"] = betting_data["Line"]
    df["over_under"] = betting_data["Over/Under"]
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
    
    print(df.columns)
    df = calc_opp_stat_diff(df, "fg")
    df = calc_opp_stat_diff(df, "fga")
    df = calc_opp_stat_diff(df, "fg%")
    df = calc_opp_stat_diff(df, "3p")
    df = calc_opp_stat_diff(df, "3pa")
    df = calc_opp_stat_diff(df, "3p%")
    df = calc_opp_stat_diff(df, "ft")
    df = calc_opp_stat_diff(df, "fta")
    df = calc_opp_stat_diff(df, "ft%")
    df = calc_opp_stat_diff(df, "orb")
    df = calc_opp_stat_diff(df, "drb")
    df = calc_opp_stat_diff(df, "trb")
    df = calc_opp_stat_diff(df, "ast")
    df = calc_opp_stat_diff(df, "stl")
    df = calc_opp_stat_diff(df, "blk")
    df = calc_opp_stat_diff(df, "tov")
    df = calc_opp_stat_diff(df, "pf")

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

    df = calc_opp_stat_avg(df, "fg")
    df = calc_opp_stat_avg(df, "fga")
    df = calc_opp_stat_avg(df, "fg%")
    df = calc_opp_stat_avg(df, "3p")
    df = calc_opp_stat_avg(df, "3pa")
    df = calc_opp_stat_avg(df, "3p%")
    df = calc_opp_stat_avg(df, "ft")
    df = calc_opp_stat_avg(df, "fta")
    df = calc_opp_stat_avg(df, "ft%")
    df = calc_opp_stat_avg(df, "orb")
    df = calc_opp_stat_avg(df, "drb")
    df = calc_opp_stat_avg(df, "trb")
    df = calc_opp_stat_avg(df, "ast")
    df = calc_opp_stat_avg(df, "stl")
    df = calc_opp_stat_avg(df, "blk")
    df = calc_opp_stat_avg(df, "tov")
    df = calc_opp_stat_avg(df, "pf")
    
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
        "opp_ast_avg", "opp_stl_avg", "opp_blk_avg", "opp_tov_avg", "opp_pf_avg"
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

from sklearn.preprocessing import LabelEncoder

def create_lstm_dataset(sequences, targets):
    logging.info("Creating LSTM dataset...")
    X = np.array([seq for seq in sequences])
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit and transform the targets
    y = label_encoder.fit_transform(targets)
    
    logging.info("LSTM dataset created.")
    return X, y

def create_xgb_dataset(data):
    logging.info("Creating XGBoost dataset...")
    X = data.drop(columns=['game_id', 'winner'])
    y = data['winner']
    groups = data.groupby('game_id').size().to_frame('group')['group']
    logging.info("XGBoost dataset created.")
    return X, y, groups

def create_lstm_model(input_shape, num_classes):
    logging.info("Creating LSTM model...")
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = LSTM(50, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)  # Increased dropout rate
    x = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
    attention = tf.keras.layers.AdditiveAttention(name='attention_weight')
    attention_result = attention([x, x])
    x = Multiply()([x, attention_result])
    x = Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Flatten()(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    loss = 'categorical_crossentropy'
    
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    logging.info("LSTM model created.")
    return model

def create_xgboost_models():
    logging.info("Creating XGBoost models...")
    
    # Learning to Rank model
    ltr_model = xgb.XGBRanker(objective='rank:ndcg', eval_metric='ndcg@10', tree_method='hist', device='cuda')
    ltr_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
        'reg_lambda': [0, 0.1, 0.5],  # L2 regularization
    }
    ltr_grid_search = GridSearchCV(ltr_model, ltr_param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    
    # Quantile Regression model
    qr_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda')
    qr_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
        'reg_lambda': [0, 0.1, 0.5],  # L2 regularization
    }
    qr_grid_search = GridSearchCV(qr_model, qr_param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    
    logging.info("XGBoost models created.")
    return ltr_grid_search, qr_grid_search

def blend_predictions(pred_lstm, pred_xgb):
    logging.info("Blending predictions...")
    blended_preds = 0.5 * pred_lstm + 0.5 * pred_xgb
    logging.info("Predictions blended.")
    return blended_preds

def train_meta_model(X, y):
    logging.info("Training meta-model...")
    meta_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    meta_model.fit(X, y)
    logging.info("Meta-model trained.")
    return meta_model

def run_pipeline(historical_data, odds_data):
    logging.info("Running pipeline...")
    
    logging.info("Preprocessing and preparing data...")
    (X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test), (X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test, groups_xgb_train, groups_xgb_test), selected_columns = prepare_data(historical_data, odds_data)
    
    logging.info("Standardizing XGBoost data...")
    scaler = StandardScaler()
    X_xgb_train_scaled = scaler.fit_transform(X_xgb_train)
    X_xgb_test_scaled = scaler.transform(X_xgb_test)

    logging.info("Performing feature selection...")
    from sklearn.feature_selection import RFE
    selector = RFE(estimator=xgb.XGBClassifier(n_estimators=100), n_features_to_select=20, step=1)
    selector.fit(X_xgb_train_scaled, y_xgb_train)
    X_xgb_train_selected = selector.transform(X_xgb_train_scaled)
    X_xgb_test_selected = selector.transform(X_xgb_test_scaled)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_checkpoint = ModelCheckpoint(f'best_lstm_model_{timestamp}.keras', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    tensorboard = TensorBoard(log_dir='./logs')
    csv_logger = CSVLogger('training_log.csv')
    callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(os.path.join(model_dir, f'best_lstm_model_{timestamp}.keras'),
                                    save_best_only=True, monitor='val_loss', mode='min')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    logging.info("Creating LSTM model...")
    num_classes = y_lstm_train.shape[1]
    lstm_model = create_lstm_model(X_lstm_train.shape[1:], num_classes)
    
    logging.info("Creating XGBoost models...")
    ltr_model, qr_model = create_xgboost_models()

    print("X_lstm_train dtype:", X_lstm_train.dtype)
    print("y_lstm_train dtype:", y_lstm_train.dtype)
    logging.info("Training LSTM model...")
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=callbacks_list)
    logging.info("LSTM model trained.")

    best_lstm_model_path = os.path.join(model_dir, f'best_lstm_model_{timestamp}.keras')
    if os.path.exists(best_lstm_model_path):
        logging.info("Loading the best LSTM model...")
        lstm_model = keras.models.load_model(best_lstm_model_path)
    else:
        logging.info("Best LSTM model not found. Using the trained model.")
        logging.info("Generating LSTM features for XGBoost...")
        lstm_train_features = lstm_model.predict(X_lstm_train)
        lstm_test_features = lstm_model.predict(X_lstm_test)

    # Add a small epsilon value to the input of the softmax function
    epsilon = 1e-8
    lstm_train_features = np.clip(lstm_train_features, epsilon, 1 - epsilon)
    lstm_test_features = np.clip(lstm_test_features, epsilon, 1 - epsilon)

    logging.info("Combining features...")
    X_train_combined = np.hstack((X_xgb_train_selected, lstm_train_features))
    X_test_combined = np.hstack((X_xgb_test_selected, lstm_test_features))

    logging.info("Training Learning to Rank model...")
    ltr_model.fit(X_train_combined, y_xgb_train, group=groups_xgb_train)
    best_ltr_model = ltr_model.best_estimator_
    logging.info("Learning to Rank model trained.")

    logging.info("Training Quantile Regression model...")
    qr_model.fit(X_train_combined, y_xgb_train)
    best_qr_model = qr_model.best_estimator_
    logging.info("Quantile Regression model trained.")

    # Save the best Learning to Rank model
    logging.info("Saving the best Learning to Rank model...")
    with open(f'best_ltr_model_{timestamp}.pkl', 'wb') as file:
        pickle.dump(best_ltr_model, file)

    # Save the best Quantile Regression model
    logging.info("Saving the best Quantile Regression model...")
    with open(f'best_qr_model_{timestamp}.pkl', 'wb') as file:
        pickle.dump(best_qr_model, file)

    logging.info("Making predictions...")
    X_lstm_test_reshaped = X_lstm_test.reshape(X_lstm_test.shape[0], X_lstm_test.shape[1], -1)
    pred_lstm = lstm_model.predict(X_lstm_test_reshaped)
    pred_ltr = best_ltr_model.predict(X_test_combined)
    pred_qr = best_qr_model.predict(X_test_combined)
    logging.info("Predictions made.")

    logging.info("Setting up meta-model...")
    X_lstm_test_reshaped = X_lstm_test.reshape(X_lstm_test.shape[0], -1)
    meta_X_train = np.vstack([X_lstm_test_reshaped, pred_ltr, pred_qr]).T
    meta_y_train = np.argmax(y_lstm_test, axis=1)

    meta_models = [
        xgb.XGBClassifier(n_estimators=150, random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(kernel='linear', probability=True)
    ]

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    meta_preds = []
    logging.info("Performing time series cross-validation...")
    for train_index, test_index in tscv.split(meta_X_train):
        X_train, X_test = meta_X_train[train_index], meta_X_train[test_index]
        y_train, y_test = meta_y_train[train_index], meta_y_train[test_index]

        meta_models = [
            xgb.XGBClassifier(n_estimators=150, random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42),
            SVC(kernel='linear', probability=True)
        ]

        fold_preds = []
        for model in meta_models:
            logging.info(f"Training meta-model: {type(model).__name__}")
            model.fit(X_train, y_train)
            logging.info(f"Meta-model trained: {type(model).__name__}")
            fold_preds.append(model.predict_proba(X_test)[:, 1])

        meta_preds.append(np.mean(fold_preds, axis=0))

    logging.info("Averaging predictions across all folds...")
    meta_preds = np.array(meta_preds)
    pred_final = np.mean(meta_preds, axis=0)

    logging.info("Evaluating final predictions...")
    final_acc = accuracy_score(meta_y_train, pred_final.round())
    logging.info(f"Meta-Model Accuracy: {final_acc}")

    logging.info("Finding the best meta-model...")
    best_meta_model = None
    best_meta_accuracy = 0.0

    for i, model in enumerate(meta_models):
        meta_preds_model = meta_preds[i]
        meta_accuracy = accuracy_score(meta_y_train, np.round(meta_preds_model))
        
        if meta_accuracy > best_meta_accuracy:
            best_meta_model = model
            best_meta_accuracy = meta_accuracy

    logging.info(f"Best meta-model: {type(best_meta_model).__name__}")
    logging.info(f"Best meta-model accuracy: {best_meta_accuracy}")

    # Save the best meta-model
    logging.info("Saving the best meta-model...")
    with open(f'best_meta_model_{timestamp}.pkl', 'wb') as file:
        pickle.dump(best_meta_model, file)

    logging.info("Pipeline completed.")
    return best_ltr_model, best_qr_model, lstm_model, best_meta_model

odds_data = pd.read_csv("combined_od_v1.csv")
historical_data = pd.read_csv("2012_to_2024_data.csv")

logging.info("Starting pipeline...")
run_pipeline(historical_data, odds_data)