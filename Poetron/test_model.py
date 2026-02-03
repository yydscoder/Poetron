from src.load_kaggle_model import load_kaggle_model

try:
    model = load_kaggle_model('./models/poetry_model')
    model.load_tokenizer()
    print('Model loaded successfully!')
    
    # Try to generate a simple test
    poems = model.generate_poem(prompt='<POETRY> test', max_length=20)
    print('Generation test:', poems[0][:100] if poems else 'No output')
except Exception as e:
    print(f'Model loading failed: {e}')
    import traceback
    traceback.print_exc()
