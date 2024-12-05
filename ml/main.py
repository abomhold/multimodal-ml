from sklearn.model_selection import KFold
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, optimizers, callbacks, metrics
import numpy as np
import pandas as pd
import logging
from transformers import TFAutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Constants
SEQUENCE_LENGTH = 100
TRAIT_NAMES = ['ope', 'con', 'ext', 'agr', 'neu']
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 2e-5
K_FOLDS = 5
TRANSFORMER_MODEL_NAME = "bert-base-uncased"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(cleaned_path, liwc_path):
    """
    Load and preprocess data.
    """
    data = pd.read_csv(cleaned_path).set_index('userid')
    liwc = pd.read_csv(liwc_path).set_index('userId')

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    tokenized = tokenizer(
        data['text'].fillna("").tolist(),
        padding='max_length',
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors='np'
    )
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    numeric_cols = liwc.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    scaled_ling = scaler.fit_transform(liwc[numeric_cols].fillna(liwc.median()))

    y = (data[TRAIT_NAMES].values - 1.0) / 4.0  # Scale to [0, 1]

    return input_ids, attention_mask, scaled_ling, y

class TransformerBlock(layers.Layer):
    def __init__(self, transformer_model_name, **kwargs):
        super().__init__(**kwargs)
        self.transformer = TFAutoModel.from_pretrained(transformer_model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

def build_model(n_linguistic_features):
    """
    Build a transformer-based model.
    """
    transformer_layer = TransformerBlock(TRANSFORMER_MODEL_NAME)

    input_ids = layers.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')
    ling_input = layers.Input(shape=(n_linguistic_features,), name='ling_input')

    transformer_output = transformer_layer([input_ids, attention_mask])
    pooled_output = layers.GlobalMaxPooling1D()(transformer_output)
    ling_output = layers.Dense(64, activation='relu')(ling_input)

    combined = layers.Concatenate()([pooled_output, ling_output])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)

    outputs = layers.Dense(len(TRAIT_NAMES), activation='sigmoid')(combined)
    model = Model(inputs=[input_ids, attention_mask, ling_input], outputs=outputs)

    metrics = {
        trait: [
            keras.metrics.MeanSquaredError(name=f'{trait}_mse'),
            keras.metrics.MeanAbsoluteError(name=f'{trait}_mae'),
            keras.metrics.RootMeanSquaredError(name=f'{trait}_rmse')
        ] for trait in TRAIT_NAMES
    }


    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['accuracy'] + [item for sublist in metrics.values() for item in sublist])
    return model

def train_with_cross_validation(input_ids, attention_mask, scaled_ling, y):
    """
    Train and evaluate the model using K-Fold Cross-Validation.
    """
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_no = 1

    for train_idx, val_idx in kfold.split(input_ids):
        logging.info(f"Training fold {fold_no}/{K_FOLDS}...")
        X_train = [input_ids[train_idx], attention_mask[train_idx], scaled_ling[train_idx]]
        X_val = [input_ids[val_idx], attention_mask[val_idx], scaled_ling[val_idx]]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(scaled_ling.shape[1])
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )

        val_loss, val_mse, val_mae, val_acc = model.evaluate(X_val, y_val, verbose=1)
        logging.info(f"Fold {fold_no} - Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_acc:.4f}")
        # Evaluate metrics for each trait
        logging.info("Evaluating model performance on test set:")
        for i, trait in enumerate(TRAIT_NAMES):
            X_test_ids, X_test_mask, X_test_ling, y_test = X_val[0], X_val[1], X_val[2], y_val
            y_test_rescaled = y_test * 4.0 + 1.0
            y_pred = model.predict([X_test_ids, X_test_mask, X_test_ling])
            y_pred_rescaled = y_pred * 4.0 + 1.0
            mse = mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
            r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
            mae = mean_absolute_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
            logging.info(f"{trait.upper()} - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            fold_no += 1

def main():
    input_ids, attention_mask, scaled_ling, y = load_data('./cleaned.csv', './data/training/LIWC/LIWC.csv')
    train_with_cross_validation(input_ids, attention_mask, scaled_ling, y)

if __name__ == "__main__":
    main()
