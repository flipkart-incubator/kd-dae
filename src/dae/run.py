from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    IntervalStrategy,
    TrainingArguments,
)
from typing import Optional, Dict
from tqdm import tqdm

import torch
from typing import List
import json
import random

import warnings

from src.config import TrainingConfig
from dataset import DataCollatorForCompletionOnlyLM, CDataset
from peft import LoraConfig, TaskType, get_peft_model

from trainer import DAETrainer
import os
from os.path import join

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    experiment_name: str = field(
        metadata={"help": "experiment name"},
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "local rank of process"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
local_rank = script_args.local_rank

experiment_config_path = f"./config/{script_args.experiment_name}.yml"
config = TrainingConfig.from_file(experiment_config_path)

config.output_dir = join(
    config.output_dir,
    f"{config.base_model_path.rstrip('/').split('/')[-1]}-dae{config.experiment_name}",
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

if not tokenizer.pad_token_id:
    print("adding pad token..")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

tokenizer.padding_side = "right"

"""
Set up data collator and load the data
"""
# Data Collator
collator = DataCollatorForCompletionOnlyLM(
    max_seq_len=config.seq_length,
    instruction_template=config.instruction_key,
    response_template=config.response_key,
    tokenizer=tokenizer,
    mlm=False,
    dataset_text_field="text",
)


# train_data, eval_data: list of strings
train_data = pickle.load(open(config.training_data_path, "rb"))
eval_data = pickle.load(open(config.eval_data_path, "rb"))

train_data = CDataset(prompts=train_data)
eval_data = CDataset(prompts=eval_data)

print(f"Training Data Size: {len(train_data)}")
print(f"Evaluation Data Size: {len(eval_data)}")

total_steps_per_epoch = (len(train_data) + len(eval_data)) / (
    config.train_batch_size * 8 * config.gradient_accumulation_steps
)
save_eval_steps = int(0.25 * total_steps_per_epoch)
print(f"{save_eval_steps} - Save and Evaluation Steps")

"""
Set up training arguments and begin training
"""
training_args = TrainingArguments(
    remove_unused_columns=False,
    group_by_length=False,
    output_dir=config.output_dir,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    gradient_checkpointing=config.gradient_checkpointing,
    evaluation_strategy="no",  # IntervalStrategy.STEPS
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    logging_steps=config.logging_steps,
    deepspeed=config.deepspeed,
    max_steps=config.max_steps,
    load_best_model_at_end=False,
    lr_scheduler_type=getattr(config, "lr_scheduler", "cosine"),
    report_to="tensorboard",
    local_rank=local_rank,
    save_strategy="steps",
    save_total_limit=1,
    weight_decay=config.weight_decay,
    warmup_steps=config.warmup_steps,
    save_steps=save_eval_steps,
    bf16=True,
)

"""
Load the model
"""
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_path,
    use_cache=not config.gradient_checkpointing,
    torch_dtype=torch.bfloat16,
)

target_model = AutoModelForCausalLM.from_pretrained(
    config.target_model_path,
    use_cache=not config.gradient_checkpointing,
    torch_dtype=torch.bfloat16,
)

domain_model = AutoModelForCausalLM.from_pretrained(
    config.domain_model_path,
    use_cache=not config.gradient_checkpointing,
    torch_dtype=torch.bfloat16,
)

# Define the LoraConfig
if config.use_peft:
    peft_config = LoraConfig(
        r=config.peft_lora_r,
        lora_alpha=config.peft_lora_alpha,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.15,
    )

    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
else:
    peft_config = None


trainer = DAETrainer(
    temperature=config.temperature,
    lambda_param=config.lambda_param,
    ref_model_domain=domain_model,
    ref_model=target_model,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=collator,
)

trainer.model.resize_token_embeddings(len(tokenizer))
trainer.ref_model.resize_token_embeddings(len(tokenizer))
trainer.ref_model_domain.resize_token_embeddings(len(tokenizer))

trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

# Save the model
if config.use_peft:
    trainer.model = trainer.model.merge_and_unload()

trainer.save_model(config.output_dir)
