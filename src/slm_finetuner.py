import torch
from datasets import Dataset
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def finetune_slm(model_id="Qwen/Qwen2.5-1.5B-Instruct", data_path="data/bfsi_dataset.json", output_dir="models/bfsi-slm-lora"):
    print(f"Loading dataset from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dataset = Dataset.from_list(data)
    
    # Configure quantization for hardware efficiency (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading Base Model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # Formalize formatting function for SFT (Alpaca style)
    def formatting_func(example):
        text = f"<|im_start|>system\nYou are a helpful and compliant BFSI Assistant.<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return [text]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100, # Testing purposes, increase for full convergence
        learning_rate=2e-4,
        logging_steps=10,
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=50,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned LoRA adapters to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully.")

if __name__ == "__main__":
    finetune_slm()
