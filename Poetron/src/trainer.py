"""
Model training functionality for the Poetry Generator
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Optional

from data_preprocessing import preprocess_poetry_data, split_into_training_chunks, add_style_tokens


class PoetryDataset(Dataset):
    """Custom dataset for poetry training."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


def train_model(
    data_path: str, 
    epochs: int = 3, 
    model_name: str = 'gpt2', 
    output_dir: str = './models', 
    max_length: int = 512,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    sample_size: int = None
):
    """
    Train or fine-tune a GPT-2 model on poetry data provided by kaggle.
    
    Args:
        data_path (str): Path to the training data file
        epochs (int): Number of training epochs
        model_name (str): Base model name for training
        output_dir (str): Directory to save the trained model
        max_length (int): Maximum sequence length
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        sample_size (int): Limit training to first N poems (optional)
        
    Returns:
        str: Path to the trained model
    """
    print(f"Loading tokenizer and model: {model_name}")
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Preprocess the training data
    print("Preprocessing training data...")
    poems = preprocess_poetry_data(data_path)
    
    # Sample if specified
    if sample_size:
        poems = poems[:sample_size]
        print(f"Limited to {sample_size} samples")
    
    # Add style tokens to differentiate poem types during training
    # For now, we'll add a generic poetry token, but this could be expanded
    poems_with_tokens = add_style_tokens(poems, "POETRY")
    
    # Split into training chunks
    training_chunks = split_into_training_chunks(poems_with_tokens, max_length)
    
    print(f"Loaded {len(training_chunks)} training samples")
    
    # Create dataset
    dataset = PoetryDataset(training_chunks, tokenizer, max_length)
    
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=learning_rate,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        report_to=[],  # Disable reporting to external services
        remove_unused_columns=False,  # Important for custom datasets especially small ones where performance issues may be present
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print("Starting training!")
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model_output_dir = f"{output_dir}/poetry_model_finetuned"
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    # Save training configuration
    config = {
        'model_name': model_name,
        'epochs_trained': epochs,
        'max_length': max_length,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'training_data_path': data_path
    }
    
    config_path = f"{model_output_dir}/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model trained and saved to {model_output_dir}")
    
    return model_output_dir


def load_trained_model(model_path: str):
    """
    Load a previously trained model.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        Tuple: (model, tokenizer)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer