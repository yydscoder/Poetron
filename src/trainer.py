import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# ... (Keep your PoetryDataset and imports the same)

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
    print(f"Loading tokenizer: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Configuration for 4-bit Quantization (Saves VRAM for Dual T4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # T4 preferred
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model with Dual-GPU sharding...")
    # 2. Load model with device_map="auto" to split across both T4s
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", 
        trust_remote_code=True
    )

    # 3. Prepare for LoRA Fine-tuning
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["c_attn"], # Specific to GPT-2
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data Processing (Keep your existing logic) ---
    print("Preprocessing training data...")
    poems = preprocess_poetry_data(data_path)
    if sample_size:
        poems = poems[:sample_size]
    
    poems_with_tokens = add_style_tokens(poems, "POETRY")
    training_chunks = split_into_training_chunks(poems_with_tokens, max_length)
    dataset = PoetryDataset(training_chunks, tokenizer, max_length)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Optimized Training Arguments for T4
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,   # Increases effective batch size without OOM
        learning_rate=learning_rate,
        fp16=True,                       # Hardware acceleration for T4
        logging_steps=10,
        max_grad_norm=0.3,               # Stability for quantized training
        optim="paged_adamw_32bit",       # Offloads optimizer states to CPU if needed
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting training on Dual T4!")
    trainer.train()

    # 5. Save the PEFT Adapter
    model_output_dir = f"{output_dir}/poetry_model_finetuned"
    trainer.model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    print(f"Model trained and saved to {model_output_dir}")
    return model_output_dir