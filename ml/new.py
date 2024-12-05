import keras
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Constants
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
LEARNING_RATE = 2e-5
DEFAULT_K_FOLDS = 5
DEFAULT_TRANSFORMER_NAME = "bert-base-uncased"
TRAIT_NAMES = ['ope', 'con', 'ext', 'agr', 'neu']
DEFAULT_RANDOM_STATE = 42
DEFAULT_SCALER_MIN = 1.0
DEFAULT_SCALER_MAX = 4.0
EARLY_STOP_PATIENCE = 2
DENSE_LAYER_UNITS_LIWC = 64
DENSE_LAYER_UNITS_COMBINED = 128
DROPOUT_RATE = 0.3

@dataclass
class ModelConfig:
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = LEARNING_RATE
    k_folds: int = DEFAULT_K_FOLDS
    transformer_name: str = DEFAULT_TRANSFORMER_NAME
    trait_names: List[str] = None

    def __post_init__(self):
        self.trait_names = TRAIT_NAMES

class TransformerBlock(layers.Layer):
    def __init__(self, transformer_name, trainable=True):
        super().__init__(trainable=trainable)
        self.bert = TFAutoModel.from_pretrained(transformer_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert(input_ids, attention_mask=attention_mask, training=self.trainable)[0]

class PersonalityPredictor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.transformer_name)
        logging.basicConfig(level=logging.INFO)

    def load_data(self, text_path: str, liwc_path: str) -> Tuple:
        """Load and preprocess text and LIWC features."""
        # Load data
        text_data = pd.read_csv(text_path).set_index('userid')
        liwc_data = pd.read_csv(liwc_path).set_index('userId')

        # Tokenize text
        tokenized = self.tokenizer(
            text_data['text'].fillna("").tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.config.sequence_length,
            return_tensors='np'
        )

        # Scale LIWC features
        numeric_cols = liwc_data.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler()
        scaled_liwc = scaler.fit_transform(liwc_data[numeric_cols].fillna(liwc_data.median()))

        # Scale personality traits to [0, 1]
        y = (text_data[self.config.trait_names].values - DEFAULT_SCALER_MIN) / (DEFAULT_SCALER_MAX - DEFAULT_SCALER_MIN)

        return tokenized['input_ids'], tokenized['attention_mask'], scaled_liwc, y

    def build_model(self, n_liwc_features: int) -> Model:
        """Build the neural network model."""
        # Input layers
        input_ids = layers.Input(shape=(self.config.sequence_length,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(self.config.sequence_length,), dtype=tf.int32, name='attention_mask')
        liwc_input = layers.Input(shape=(n_liwc_features,), name='liwc_input')

        # BERT layer with custom wrapper
        transformer_block = TransformerBlock(self.config.transformer_name)
        bert_output = transformer_block([input_ids, attention_mask])
        text_features = layers.GlobalMaxPooling1D()(bert_output)

        # LIWC features processing
        liwc_features = layers.Dense(DENSE_LAYER_UNITS_LIWC, activation='relu')(liwc_input)

        # Combine features
        combined = layers.Concatenate()([text_features, liwc_features])
        x = layers.Dense(DENSE_LAYER_UNITS_COMBINED, activation='relu')(combined)
        x = layers.Dropout(DROPOUT_RATE)(x)
        outputs = layers.Dense(len(self.config.trait_names), activation='sigmoid')(x)

        model = Model(
            inputs={'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'liwc_input': liwc_input},
            outputs=outputs
        )

        metrics = {
            trait: [
                keras.metrics.MeanSquaredError(name=f'{trait}_mse'),
                keras.metrics.MeanAbsoluteError(name=f'{trait}_mae'),
                keras.metrics.RootMeanSquaredError(name=f'{trait}_rmse')
            ] for trait in TRAIT_NAMES
        }


        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mse',
              metrics=['accuracy'] + [item for sublist in metrics.values() for item in sublist])

        return model
        # Increased learning rate
        # model.compile(
        #     optimizer=keras.optimizers.AdamW(
        #         learning_rate=0.001,  # Increased from 0.0002
        #         weight_decay=0.01
        #     ),
        #     loss={trait: 'mse' for trait in },
        #     metrics={trait: ['mae', 'mse'] for trait in trait_names}
        # )
        # # model.compile(
        # #     optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
        #     loss='mse',
        #     metrics=['mae', 'mse', 'accuracy']
        # )

    def train_and_evaluate(self, input_ids: np.ndarray, attention_mask: np.ndarray,
                           liwc_features: np.ndarray, y: np.ndarray) -> Dict:
        """Train model with cross-validation and return metrics."""
        kfold = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(input_ids), 1):
            logging.info(f"Training fold {fold}/{self.config.k_folds}")

            # Prepare data
            train_data = {
                'input_ids': input_ids[train_idx],
                'attention_mask': attention_mask[train_idx],
                'liwc_input': liwc_features[train_idx]
            }
            val_data = {
                'input_ids': input_ids[val_idx],
                'attention_mask': attention_mask[val_idx],
                'liwc_input': liwc_features[val_idx]
            }
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = self.build_model(liwc_features.shape[1])
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True
            )

            model.fit(
                train_data, y_train,
                validation_data=(val_data, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=[early_stop],
                verbose=1
            )

            # Evaluate
            y_pred = model.predict(val_data)
            fold_results = self._calculate_metrics(y_val, y_pred)
            results.append(fold_results)

            logging.info(f"Fold {fold} Results:")
            for trait, metrics in fold_results.items():
                logging.info(f"{trait}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

        return self._aggregate_results(results)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics for each trait."""
        # Rescale predictions back to original scale
        y_true_rescaled = y_true * (DEFAULT_SCALER_MAX - DEFAULT_SCALER_MIN) + DEFAULT_SCALER_MIN
        y_pred_rescaled = y_pred * (DEFAULT_SCALER_MAX - DEFAULT_SCALER_MIN) + DEFAULT_SCALER_MIN

        results = {}
        for i, trait in enumerate(self.config.trait_names):
            results[trait] = {
                'mse': mean_squared_error(y_true_rescaled[:, i], y_pred_rescaled[:, i]),
                'r2': r2_score(y_true_rescaled[:, i], y_pred_rescaled[:, i]),
                'mae': mean_absolute_error(y_true_rescaled[:, i], y_pred_rescaled[:, i])
            }
        return results

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all folds."""
        final_results = {}
        for trait in self.config.trait_names:
            trait_metrics = {
                'mse': np.mean([fold[trait]['mse'] for fold in results]),
                'r2': np.mean([fold[trait]['r2'] for fold in results]),
                'mae': np.mean([fold[trait]['mae'] for fold in results])
            }
            final_results[trait] = trait_metrics
        return final_results

def main():
    config = ModelConfig()
    predictor = PersonalityPredictor(config)

    # Load and preprocess data
    input_ids, attention_mask, liwc_features, y = predictor.load_data(
        './cleaned.csv',
        './data/training/LIWC/LIWC.csv'
    )

    # Train and evaluate
    results = predictor.train_and_evaluate(input_ids, attention_mask, liwc_features, y)

    # Print final results
    print("\nFinal Results (averaged across folds):")
    for trait, metrics in results.items():
        print(f"\n{trait.upper()}:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
