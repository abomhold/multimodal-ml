# Configure logging
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.python.ops.metrics_impl import precision
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and Hyperparameters
SEQUENCE_LENGTH = 200
TRAIT_NAMES = ['ope', 'con', 'ext', 'agr', 'neu']
EMBEDDING_DIM = 128
LSTM_UNITS = 8
DENSE_UNITS = 8
BATCH_SIZE = 32
EPOCHS = 300
NUM_FOLDS = 2
RANDOM_STATE = 42
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-6
SCALE_CON_MIN = 1.25
SCALE_CON_RANGE = 3.75
SCALE_OTHER_MIN = 1.0
SCALE_OTHER_RANGE = 4.0
CLIP_MIN = 1.0
CLIP_MAX = 5.0


def load_and_preprocess(cleaned_path: str, liwc_path: str):
    """Load data, clean 'words' column, preprocess features and targets."""
    data = pd.read_csv(cleaned_path)
    liwc_profile = pd.read_csv(liwc_path)

    data['words'] = data['words'].fillna("").astype(str)

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(
        data['words'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors='np'
    )['input_ids']

    linguistic_data = liwc_profile.iloc[:, 1:]
    linguistic_data = linguistic_data.drop(columns=['Seg'])
    scaler = MinMaxScaler()
    scaled_ling = scaler.fit_transform(linguistic_data)

    y = data[TRAIT_NAMES].values
    y_scaled = np.zeros_like(y, dtype=np.float32)
    scaling_factors = {}
    for i, trait in enumerate(TRAIT_NAMES):
        if trait == 'con':
            y_scaled[:, i] = (y[:, i] - SCALE_CON_MIN) / SCALE_CON_RANGE
            scaling_factors[trait] = {'min': SCALE_CON_MIN, 'scale': SCALE_CON_RANGE}
        else:
            y_scaled[:, i] = (y[:, i] - SCALE_OTHER_MIN) / SCALE_OTHER_RANGE
            scaling_factors[trait] = {'min': SCALE_OTHER_MIN, 'scale': SCALE_OTHER_RANGE}

    return tokens, scaled_ling, y_scaled, scaling_factors

def build_model(vocab_size: int, n_linguistic_features: int) -> Model:
    """Build and compile the multi-input model."""
    text_input = layers.Input(shape=(SEQUENCE_LENGTH,), name='text_input')
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name='embedding')(text_input)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, name='lstm'))(x)
    x = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense_text')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    ling_input = layers.Input(shape=(n_linguistic_features,), name='ling_input')
    y = layers.BatchNormalization(name='batch_norm')(ling_input)
    y = layers.Dense(DENSE_UNITS, activation='relu', name='dense_ling')(y)
    y = layers.Dropout(DROPOUT_RATE)(y)

    combined = layers.Concatenate(name='concatenate')([x, y])
    combined = layers.Dense(DENSE_UNITS * 2, activation='relu', name='dense_combined')(combined)
    combined = layers.Dropout(DROPOUT_RATE)(combined)
    combined = layers.Dense(DENSE_UNITS, activation='relu', name='dense_final')(combined)

    outputs = [layers.Dense(1, activation='linear', name=trait)(combined) for trait in TRAIT_NAMES]

    model = Model(inputs=[text_input, ling_input], outputs=outputs)
    metrics = {
        trait: [
            keras.metrics.MeanSquaredError(name=f'{trait}'),
            keras.metrics.MeanAbsoluteError(name=f'{trait}'),
            keras.metrics.RootMeanSquaredError(name=f'{trait}'),
        ] for trait in TRAIT_NAMES
    }

    logging.log(logging.INFO, f'metrics: {metrics}')

    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=metrics)
    return model

def train_and_evaluate(tokens, scaled_ling, y_scaled, scaling_factors):
    """Train and evaluate the model using K-Fold cross-validation with stock metrics."""
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {trait: {'correlation': [], 'mae': []} for trait in TRAIT_NAMES}

    for fold, (train_idx, val_idx) in enumerate(kf.split(tokens)):
        logging.info(f"\nFold {fold + 1}/{NUM_FOLDS}")

        # Build and compile model
        model = build_model(
            vocab_size=AutoTokenizer.from_pretrained("bert-base-uncased").vocab_size,
            n_linguistic_features=scaled_ling.shape[1]
        )

        # Early stopping to prevent overfitting
        early_stop_loss = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Reduce learning rate for stable convergence
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        # Save best model per fold
        checkpoint = callbacks.ModelCheckpoint(
            f'model_fold_{fold + 1}.keras', save_best_only=True, monitor='val_loss', mode='min'
        )

        callback_list = [early_stop_loss, reduce_lr, checkpoint]

        # Train model
        model.fit(
            [tokens[train_idx], scaled_ling[train_idx]],
            [y_scaled[train_idx, i] for i in range(y_scaled.shape[1])],
            validation_data=(
                [tokens[val_idx], scaled_ling[val_idx]],
                [y_scaled[val_idx, i] for i in range(y_scaled.shape[1])]
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callback_list,
            verbose=1
        )

        # Load best weights and evaluate
        model.load_weights(f'model_fold_{fold + 1}.keras')
        predictions = np.array(model.predict([tokens[val_idx], scaled_ling[val_idx]])).squeeze().T

        # Rescale predictions and ground truth
        predictions_rescaled = np.zeros_like(predictions)
        actual_rescaled = np.zeros_like(predictions)
        for i, trait in enumerate(TRAIT_NAMES):
            predictions_rescaled[:, i] = predictions[:, i] * scaling_factors[trait]['scale'] + scaling_factors[trait][
                'min']
            actual_rescaled[:, i] = y_scaled[val_idx, i] * scaling_factors[trait]['scale'] + scaling_factors[trait][
                'min']

        # Calculate and log metrics
        for i, trait in enumerate(TRAIT_NAMES):
            pred, act = predictions_rescaled[:, i], actual_rescaled[:, i]
            fold_metrics[trait]['r2'].append(keras.metrics.R2Score(act,pred))
            fold_metrics[trait]['correlation'].append(keras.metrics.PearsonCorrelation(act,pred))
            fold_metrics[trait]['mae'].append(keras.metrics.MeanAbsoluteError(act,pred))
            fold_metrics[trait]['auc'].append(keras.metrics.AUC(act,pred))
            fold_metrics[trait]['precision'].append(keras.metrics.Precision(act,pred))
            logging.info(f"{trait.upper()} | " 
                         f"R2: {fold_metrics[trait]['r2'][-1]:.4f} ± {fold_metrics[trait]['r2'][-1]:.4f} | "
                         f"Corr: {fold_metrics[trait]['correlation'][-1]:.4f} ± {fold_metrics[trait]['correlation'][-1]:.4f} | "
                         f"MAE: {fold_metrics[trait]['mae'][-1]:.4f} ± {fold_metrics[trait]['mae'][-1]:.4f} | "
                         f"AUC: {fold_metrics[trait]['auc'][-1]:.4f} ± {fold_metrics[trait]['auc'][-1]:.4f} | "
                         f"Prec: {fold_metrics[trait]['precision'][-1]:.4f} ± {fold_metrics[trait]['precision'][-1]:.4f} | "
                         )

    # Aggregate and log metrics
    logging.info("\nFinal Metrics:")
    for trait in TRAIT_NAMES:
        metrics = fold_metrics[trait]
        logging.info(f"{trait.upper()} | "
                     f"R2: {metrics['r2'][-1]:.4f} ± {metrics['r2'][-1]:.4f} | "
                     f"Corr: {metrics['correlation'][-1]:.4f} ± {metrics['correlation'][-1]:.4f} | "
                     f"MAE: {metrics['mae'][-1]:.4f} ± {metrics['mae'][-1]:.4f} | "
                     f"AUC: {metrics['auc'][-1]:.4f} ± {metrics['auc'][-1]:.4f} | "
                     f"Prec: {metrics['precision'][-1]:.4f} ± {metrics['precision'][-1]:.4f} | "
                     )


def main():
    # File paths
    CLEANED_DATA_PATH = 'text/cleaned.csv'
    LIWC_PROFILE_PATH = './data/training/LIWC/LIWC.csv'

    # Load and preprocess data
    tokens, scaled_ling, y_scaled, scaling_factors = load_and_preprocess(CLEANED_DATA_PATH, LIWC_PROFILE_PATH)

    logging.info("Data Preprocessing Completed.")

    # Train and evaluate the model
    train_and_evaluate(tokens, scaled_ling, y_scaled, scaling_factors)


if __name__ == "__main__":
    main()
