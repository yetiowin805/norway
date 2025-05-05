import os
import random
import logging
import time
import argparse  # Add this for command-line arguments
from datetime import timedelta
from typing import List, Tuple, Dict  # Add this import

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
    PreTrainedTokenizer
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal language model with LoRA and perform inference."
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="Number of samples for training (must be a positive integer)",
    )
    parser.add_argument(
        "--skip_inference", action="store_true", help="Skip inference after training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NorwAI/NorwAI-Mistral-7B-instruct",
        help="Model identifier from Hugging Face Hub",
    )
    args = parser.parse_args()

    # Validate train_size
    if args.train_size <= 0:
        parser.error("train_size must be a positive integer")

    return args


def setup_environment(local_rank: int, seed: int = 42) -> str:
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if local_rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
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
    complex_file = "/workspace/BiSECT/bisect/train.src"
    simplified_file = "/workspace/BiSECT/bisect/train.dst"

    with open(complex_file, "r") as f:
        complex_sentences = [line.strip() for line in f]
    with open(simplified_file, "r") as f:
        simplified_sentences = [line.strip() for line in f]

    if train_size < len(complex_sentences):
        complex_sentences = complex_sentences[:train_size]
        simplified_sentences = simplified_sentences[:train_size]

    inputs = [
        f"INSTRUCTION: Split the sentence below into separate sentences.\nINPUT: {complex}\nRESPONSE: "
        for complex in complex_sentences
    ]

    return Dataset.from_dict({"inputs": inputs, "targets": simplified_sentences})


def prepare_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    # Define maximum sequence length
    MAX_LENGTH = 320

    # Tokenization function
    def tokenize_function(examples):
        max_input_length = MAX_LENGTH // 2 - 1  # Adjust for EOS tokens
        max_target_length = MAX_LENGTH // 2 - 1

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
            max_length=max_target_length,
            add_special_tokens=False,
        )

        # Concatenate inputs and targets with EOS tokens and pad to MAX_LENGTH
        input_ids_batch = []
        for inp_ids, tgt_ids in zip(
            tokenized_inputs["input_ids"], tokenized_targets["input_ids"]
        ):
            ids = (
                inp_ids + [tokenizer.eos_token_id] + tgt_ids + [tokenizer.eos_token_id]
            )
            ids = ids + [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
            input_ids_batch.append(ids)

        return {"input_ids": input_ids_batch}

    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["inputs", "targets"]
    )

    # Prepare labels for training
    def prepare_labels(examples):
        input_ids = examples["input_ids"]
        labels = []

        for ids in input_ids:
            # Find the length of the tokenized input (including EOS token)
            input_length = ids.index(tokenizer.eos_token_id) + 1
            # Mask the input tokens and keep target tokens for computing loss
            label = [-100] * input_length + ids[input_length:]
            # Truncate label to response length and pad
            response_length = label.index(tokenizer.eos_token_id) + 1
            label = label[:response_length] + [-100] * (MAX_LENGTH - response_length)
            labels.append(label)

        examples["labels"] = labels
        return examples

    # Apply label preparation to the dataset
    tokenized_dataset = tokenized_dataset.map(prepare_labels, batched=True)

    return tokenized_dataset


def run_training_pipeline(max_length: int = 320):
    args = parse_arguments()
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(days=1)
    )

    MAX_LENGTH = 320

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    try:
        hf_home = setup_environment(local_rank)
        if local_rank == 0:
            print(f"HF_HOME: {hf_home}")
            print(args)

        model, tokenizer = load_model_and_tokenizer(args.model, hf_home)
        dataset = load_dataset(args.train_size)
        prepared_dataset = prepare_dataset(dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir="train/BiSECT",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=1,  # Keep original value
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=True,
            report_to="none",
            ddp_backend="nccl",
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=prepared_dataset,
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer, padding=True, return_tensors="pt"
            ),
        )

        start_time = time.time()

        # Use the specified train size
        train_dataset = prepared_dataset

        # Data collator for padding
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8,  # Optional: For performance optimization
        )

        if dist.get_rank() == 0:
            training_start_time = time.time()
            preparation_time = training_start_time - start_time
            print(f"Preparation Time: {preparation_time:.2f} seconds")

        # Train the model
        trainer.train()

        if dist.get_rank() == 0:
            end_time = time.time()
            training_time = end_time - training_start_time
            total_time = end_time - start_time

            print(f"Training Time: {training_time:.2f} seconds")
            print(f"Total Time: {total_time:.2f} seconds")

        # Perform inference if desired
        if not args.skip_inference and dist.get_rank() == 0:
            # Inference code remains the same as before
            complex_file = "/workspace/BiSECT/bisect/test.src"
            with open(complex_file, "r") as file:
                inputs_list = [line.strip() for line in file]

            # Limit the number of samples for evaluation
            # num_samples = 100
            # inputs_list = inputs_list[:num_samples]
            print(len(inputs_list))

            prompts = [
                f"INSTRUCTION: Split the sentence below into separate sentences.\nINPUT: {line}\nRESPONSE: "
                for line in inputs_list
            ]

            responses = []

            # Generate outputs
            for prompt in prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
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

            # Load expected outputs (references)
            expected_file = "/workspace/BiSECT/bisect/test.dst"
            with open(expected_file, "r") as file:
                expected_list = [line.strip() for line in file]

            # Ensure the number of references matches the number of responses
            expected_list = expected_list[: len(responses)]

            # For metrics that expect multiple references per prediction
            references = [[ref] for ref in expected_list]

            # Load evaluation metrics
            bert_score = load("bertscore")
            sari_score = load("sari")
            sacrebleu = load("sacrebleu")

            # Compute BERTScore
            bert_results = bert_score.compute(
                predictions=responses,
                references=expected_list,
                lang="en",  # Replace with the appropriate language code
            )

            # Compute SARI Score
            sari_results = sari_score.compute(
                sources=inputs_list, predictions=responses, references=references
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
        dist.destroy_process_group()


def main():
    run_training_pipeline()


if __name__ == "__main__":
    main()
