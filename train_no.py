import os
import random
import logging
import time
import argparse
from datetime import timedelta
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from evaluate import load
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal language model with LoRA and perform inference."
    )
    parser.add_argument(
        "--train_size", type=int, default=1000, help="Number of samples for training"
    )
    parser.add_argument(
        "--skip_inference", action="store_true", help="Skip inference after training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NorwAI/NorwAI-Mistral-7B-instruct",
        help="Model to use",
    )
    return parser.parse_args()


def setup_environment(local_rank: int) -> str:
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logging.basicConfig(level=logging.INFO if local_rank == 0 else logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hf_home = os.getenv("HF_HOME")
    if not hf_home:
        raise ValueError("HF_HOME environment variable is not set")
    os.makedirs(hf_home, exist_ok=True)
    return hf_home


def load_model_and_tokenizer(
    model_name: str, hf_home: str
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    access_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=access_token, cache_dir=hf_home
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=access_token,
        cache_dir=hf_home,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"]
        if "pythia" in model_name
        else ["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, lora_config), tokenizer


def load_dataset(train_size: int) -> Dataset:
    complex_file = "/workspace/nor/whole.txt"
    simplified_file = "/workspace/nor/split.txt"

    with open(complex_file, "r") as f:
        complex_sentences = [line.strip() for line in f][:train_size]
    with open(simplified_file, "r") as f:
        simplified_sentences = [line.strip() for line in f][:train_size]

    inputs = [
        f"INSTRUKSJON: Del opp setningen under i atskilte setninger. \nINNDATA: {complex}\nRESPONS: "
        for complex in complex_sentences
    ]

    return Dataset.from_dict({"inputs": inputs, "targets": simplified_sentences})


def prepare_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int = 320
) -> Dataset:
    def tokenize_function(examples):
        max_input_length = max_length // 2 - 1
        tokenized_inputs = tokenizer(
            examples["inputs"],
            padding=False,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        tokenized_targets = tokenizer(
            examples["targets"],
            padding=False,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )

        input_ids_batch = []
        for inp_ids, tgt_ids in zip(
            tokenized_inputs["input_ids"], tokenized_targets["input_ids"]
        ):
            ids = (
                inp_ids + [tokenizer.eos_token_id] + tgt_ids + [tokenizer.eos_token_id]
            )
            ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
            input_ids_batch.append(ids)

        return {"input_ids": input_ids_batch}

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["inputs", "targets"]
    )
    return tokenized_dataset.map(
        lambda examples: {
            "labels": prepare_labels(examples["input_ids"], tokenizer, max_length)
        },
        batched=True,
    )


def prepare_labels(
    input_ids: List[int], tokenizer: PreTrainedTokenizer, max_length: int
) -> List[int]:
    labels = []
    for ids in input_ids:
        input_length = ids.index(tokenizer.eos_token_id) + 1
        label = [-100] * input_length + ids[input_length:]
        response_length = label.index(tokenizer.eos_token_id) + 1
        label = label[:response_length] + [-100] * (max_length - response_length)
        labels.append(label)
    return labels


def run_training_pipeline(max_length: int = 320):
    args = parse_arguments()
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(days=1)
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    try:
        hf_home = setup_environment(local_rank)
        model, tokenizer = load_model_and_tokenizer(args.model, hf_home)

        # Load the full dataset
        dataset = load_dataset(args.train_size)

        # Create train-test split
        train_test_split = dataset.train_test_split(
            test_size=0.05, shuffle=True, seed=42
        )
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Prepare both datasets
        prepared_train_dataset = prepare_dataset(train_dataset, tokenizer)
        prepared_test_dataset = prepare_dataset(test_dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir="train/BiSECT",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=10,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="no",
            load_best_model_at_end=True,
            report_to="none",
            ddp_backend="nccl",
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=prepared_train_dataset,
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer, padding=True, return_tensors="pt"
            ),
        )

        trainer.train()

        # Perform inference if desired
        if not args.skip_inference and dist.get_rank() == 0:
            # Use the test dataset we created earlier
            test_inputs = test_dataset["inputs"]

            responses = []

            # Generate outputs
            for prompt in test_inputs:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.1,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,
                )

                generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                output_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )

                responses.append(output_text)
                print(f"Prompt:\n{prompt}\n{output_text}\n")

            # Get references from test dataset
            references = [[ref] for ref in test_dataset["targets"]]

            # Load evaluation metrics
            bert_score = load("bertscore")
            sari_score = load("sari")
            sacrebleu = load("sacrebleu")

            # Compute BERTScore
            bert_results = bert_score.compute(
                predictions=responses, references=test_dataset["targets"], lang="no"
            )

            # Compute SARI Score
            sari_results = sari_score.compute(
                sources=[
                    prompt.split("\nINNDATA: ")[1].split("\nRESPONS:")[0]
                    for prompt in test_inputs
                ],  # Extract original sentences
                predictions=responses,
                references=references,
            )

            # Compute SacreBLEU Score
            bleu_results = sacrebleu.compute(
                predictions=responses, references=references
            )

            # Print the evaluation results
            print("\nEvaluation Results:")
            print(f"BERTScore Precision: {np.mean(bert_results['precision']):.4f}")
            print(f"BERTScore Recall: {np.mean(bert_results['recall']):.4f}")
            print(f"BERTScore F1: {np.mean(bert_results['f1']):.4f}")
            print(f"SARI Score: {sari_results['sari']:.4f}")
            print(f"SacreBLEU Score: {bleu_results['score']:.4f}")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Clean up the process group
        dist.destroy_process_group()


def main():
    run_training_pipeline()


if __name__ == "__main__":
    main()
