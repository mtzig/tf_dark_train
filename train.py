import os
import yaml
import pickle
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_config(config_path):
    """Loads the training configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_prompt(sample, model_name=""):
    """
    Formats the chat history into a structured prompt for the model.
    It now returns both the full text for input and the prompt-only part for masking.
    """
    system_prompt = sample['system']
    user_prompt = sample['user']
    response = sample['response']

    if "qwen" in model_name.lower():
        # Qwen2 and Qwen3-Instruct format (ChatML)
        system_part = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        user_part = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        assistant_part = f"<|im_start|>assistant\n{response}<|im_end|>"
        
        prompt_only = system_part + user_part
        full_text = prompt_only + assistant_part
    else:  # Default to Llama 3 format
        system_part = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        user_part = f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        assistant_part = f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"

        prompt_only = system_part + user_part
        full_text = prompt_only + assistant_part
        
    return {"prompt": prompt_only, "text": full_text}

class SFTDataCollator:
    """
    A more robust data collator for SFT. It tokenizes the prompt and full text
    separately to determine the exact length of the prompt for masking.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        prompts = [ex["prompt"] for ex in examples]
        texts = [ex["text"] for ex in examples]
        
        # Tokenize the full texts to be used as model inputs
        text_tokens = self.tokenizer(
            texts, 
            padding='longest', 
            truncation=True, 
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt"
        )

        # Tokenize the prompts-only to determine their length for masking
        # We don't need padding or truncation here as we only need the length
        prompt_tokens = self.tokenizer(
            prompts,
            add_special_tokens=False # The special tokens are already in the prompt string
        )

        labels = text_tokens['input_ids'].clone()

        # Mask the prompt tokens
        for i in range(len(labels)):
            prompt_len = len(prompt_tokens['input_ids'][i])
            labels[i, :prompt_len] = -100
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        text_tokens['labels'] = labels
        return text_tokens

def main():
    # 1. Parse Command Line Arguments & Load Configuration
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Set up W&B environment variables
    os.environ["WANDB_PROJECT"] = config['wandb_project']

    # 2. Load and Prepare the Dataset
    print("Loading and preparing dataset...")
    with open(config['dataset_path'], 'rb') as f:
        data = pickle.load(f)
    
    dataset = Dataset.from_list(data)
    # Use data_seed for shuffling for reproducibility
    dataset = dataset.shuffle(seed=config.get('data_seed', 42))

    # Split dataset into training and validation sets
    num_validation_samples = config.get('num_validation_samples', 0)
    if num_validation_samples > 0 and num_validation_samples < len(dataset):
        print(f"Splitting dataset: {num_validation_samples} for validation, {len(dataset) - num_validation_samples} for training.")
        eval_dataset = dataset.select(range(num_validation_samples))
        train_dataset = dataset.select(range(num_validation_samples, len(dataset)))
    else:
        print("Using the entire dataset for training. No validation set will be used.")
        train_dataset = dataset
        eval_dataset = None

    # Map the prompt creation function
    train_dataset = train_dataset.map(lambda sample: create_prompt(sample, model_name=config.get('model_name', '')))
    if eval_dataset:
        eval_dataset = eval_dataset.map(lambda sample: create_prompt(sample, model_name=config.get('model_name', '')))
    
    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation examples: {len(eval_dataset)}")
    print("Sample prompt:\n", train_dataset[0]['text'])

    # 3. Load Tokenizer and Model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.model_max_length = config.get('max_seq_length', 1024)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    # 4. Set up LoRA
    print("Setting up LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Set up Training Arguments
    print("Setting up training arguments...")
    
    training_args_dict = {
        "output_dir": config['output_dir'],
        "num_train_epochs": config['num_train_epochs'],
        "per_device_train_batch_size": config['per_device_train_batch_size'],
        "gradient_accumulation_steps": config['gradient_accumulation_steps'],
        "learning_rate": config['learning_rate'],
        "lr_scheduler_type": config['lr_scheduler_type'],
        "warmup_ratio": config['warmup_ratio'],
        "weight_decay": config['weight_decay'],
        "optim": config['optim'],
        "bf16": config['bf16'],
        "fp16": config['fp16'],
        "logging_strategy": "steps",
        "logging_steps": config['logging_steps'],
        "save_strategy": "steps",
        "save_steps": config['save_steps'],
        "save_total_limit": config['save_total_limit'],
        "push_to_hub": config['push_to_hub'],
        "hub_model_id": config.get('hf_hub_repo_id'),
        "report_to": config['report_to'],
        "remove_unused_columns": False,
        "seed": config['seed'],
        "data_seed": config['data_seed'],
    }
    
    if eval_dataset:
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["eval_steps"] = config.get('eval_steps', config['logging_steps'])
        training_args_dict["per_device_eval_batch_size"] = config.get('per_device_eval_batch_size', config['per_device_train_batch_size'])

    training_args = TrainingArguments(**training_args_dict)

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. Start Training
    print("Starting training...")
    trainer.train()
    
    # 8. Save and Push the Final Model
    print("Saving final model...")
    final_model_path = os.path.join(config['output_dir'], "final")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    if config['push_to_hub']:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print("Model pushed to the Hub successfully.")

if __name__ == "__main__":
    main()

