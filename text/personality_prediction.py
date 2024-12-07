import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import torch.nn as nn
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and Hyperparameters
SEQUENCE_LENGTH = 200
TRAIT_NAMES = ['ope', 'con', 'ext', 'agr', 'neu']
BATCH_SIZE = 16
EPOCHS = 5
NUM_FOLDS = 2
RANDOM_STATE = 42
LEARNING_RATE = 2e-5
MODEL_NAME = "roberta-base"
SCALE_CON_MIN = 1.25
SCALE_CON_RANGE = 3.75
SCALE_OTHER_MIN = 1.0
SCALE_OTHER_RANGE = 4.0

class TextDataset(Dataset):
    def __init__(self, tokens, linguistic_features, labels=None):
        self.tokens = tokens
        self.linguistic_features = linguistic_features
        self.labels = labels
        self.attention_masks = (self.tokens != 1).astype(np.int64)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.tokens[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'linguistic_features': torch.tensor(self.linguistic_features[idx], dtype=torch.float)
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class PersonalityPredictor(nn.Module):
    def __init__(self, model_name: str, n_linguistic_features: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # Linguistic features processing
        self.linguistic_layer = nn.Sequential(
            nn.Linear(n_linguistic_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Combined features processing
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(TRAIT_NAMES))
        )

    def forward(self, input_ids, attention_mask, linguistic_features, labels=None):
        # Get BERT outputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Process linguistic features
        linguistic_output = self.linguistic_layer(linguistic_features)

        # Combine features
        combined = torch.cat((pooled_output, linguistic_output), dim=1)

        # Get predictions
        predictions = self.classifier(combined)

        # Calculate loss if training
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)
            return {"loss": loss, "logits": predictions}

        return {"logits": predictions}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    metrics = {}

    for i, trait in enumerate(TRAIT_NAMES):
        pred = predictions[:, i]
        label = labels[:, i]

        mse = np.mean((pred - label) ** 2)
        mae = np.mean(np.abs(pred - label))
        correlation = np.corrcoef(pred, label)[0, 1]

        metrics[f'{trait}_mse'] = mse
        metrics[f'{trait}_mae'] = mae
        metrics[f'{trait}_corr'] = correlation

    return metrics

def load_and_preprocess(cleaned_path: str, liwc_path: str):
    """Load and preprocess the data."""
    data = pd.read_csv(cleaned_path)
    liwc_profile = pd.read_csv(liwc_path)

    data['words'] = data['words'].fillna("").astype(str)

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(
        data['words'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors='np'
    )['input_ids']

    # Process linguistic features
    linguistic_data = liwc_profile.iloc[:, 1:]
    linguistic_data = linguistic_data.drop(columns=['Seg'])
    scaler = MinMaxScaler()
    scaled_ling = scaler.fit_transform(linguistic_data)

    # Scale personality traits
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

def save_model_for_export(model, scaled_ling, scaling_factors, tokenizer):
    """Save model and configuration for downloading."""
    os.makedirs('export', exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), 'export/model_state.pt')

    # Save configuration
    config = {
        'n_linguistic_features': scaled_ling.shape[1],
        'scaling_factors': scaling_factors,
    }
    torch.save(config, 'export/model_config.pt')

    # Save tokenizer
    tokenizer.save_pretrained('export/tokenizer')

    # Create zip file


# Modify the train_and_evaluate function
def train_and_evaluate(tokens, scaled_ling, y_scaled, scaling_factors):
    """Train and evaluate the model using K-Fold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_model = None
    best_eval_loss = float('inf')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for fold, (train_idx, val_idx) in enumerate(kf.split(tokens)):
        logging.info(f"\nFold {fold + 1}/{NUM_FOLDS}")

        # Create datasets for this fold
        train_dataset = TextDataset(tokens[train_idx], scaled_ling[train_idx], y_scaled[train_idx])
        val_dataset = TextDataset(tokens[val_idx], scaled_ling[val_idx], y_scaled[val_idx])

        # Initialize model
        model = PersonalityPredictor(MODEL_NAME, scaled_ling.shape[1]).to(device)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results_fold_{fold}',
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'./logs_fold_{fold}',
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            learning_rate=LEARNING_RATE,
            remove_unused_columns=False,
            metric_for_best_model="eval_loss"
        )

        # Train and evaluate
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Evaluate
        eval_results = trainer.evaluate()
        current_loss = eval_results['eval_loss']

        # Save best model
        if current_loss < best_eval_loss:
            best_eval_loss = current_loss
            best_model = model

        # Log results
        for trait in TRAIT_NAMES:
            logging.info(f"{trait.upper()}:")
            logging.info(f"  MSE: {eval_results[f'eval_{trait}_mse']:.4f}")
            logging.info(f"  MAE: {eval_results[f'eval_{trait}_mae']:.4f}")
            logging.info(f"  Correlation: {eval_results[f'eval_{trait}_corr']:.4f}")

    # Save the best model for download
    save_model_for_export(best_model, scaled_ling, scaling_factors, tokenizer)

def predict(df: pd.DataFrame, model_path: str = './export') -> pd.DataFrame:
    """Predict personality traits for new data."""
    # First load to CPU, then optionally move to MPS
    config = torch.load(f"{model_path}/model_config.pt", map_location='cpu')
    model_state = torch.load(f"{model_path}/model_state.pt", map_location='cpu')

    model = PersonalityPredictor(
        model_name=MODEL_NAME,
        n_linguistic_features=config['n_linguistic_features']
    )

    model.load_state_dict(model_state)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
    liwc_columns = config.get('liwc_columns', [col for col in df.columns
                                               if col not in ['userid', 'text', 'words', 'age', 'gender'] + TRAIT_NAMES])
    tokens = tokenizer(
        df['words'].fillna("").astype(str).tolist(),
        padding='max_length',
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors='np'
    )['input_ids']
    linguistic_features = df[liwc_columns].values
    dataset = TextDataset(tokens, linguistic_features)
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset).predictions
    results = pd.DataFrame(predictions, columns=TRAIT_NAMES)
    results['userid'] = df['userid']
    return results

def main():
    # Use direct paths in Colab environment
    CLEANED_DATA_PATH = './cleaned.csv'  # Files uploaded directly to Colab
    LIWC_PROFILE_PATH = './LIWC.csv'

    tokens, scaled_ling, y_scaled, scaling_factors = load_and_preprocess(CLEANED_DATA_PATH, LIWC_PROFILE_PATH)
    logging.info("Data Preprocessing Completed.")

    # train_and_evaluate(tokens, scaled_ling, y_scaled, scaling_factors)scaling_factors

if __name__ == "__main__":
    main()