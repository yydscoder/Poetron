import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from trainer import train_model

if __name__ == '__main__':
    train_model(
        data_path='../data/PoetryFoundationData_small.csv',
        epochs=1,
        model_name='gpt2',
        output_dir='../../models/poetry_model',
        max_length=128,
        batch_size=8,
        learning_rate=5e-5
    )
    print("Training complete!")
