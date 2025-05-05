import os
import random
import logging
import time
import argparse
from datetime import timedelta
from typing import List, Tuple, Dict
import json
import  datetime
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
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
import mlflow

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal language model with LoRA and perform inference."
    )
    parser.add_argument(
        "--train_size", type=int, default=1000, help="Number of samples for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
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
        model_name, token=access_token, cache_dir=hf_home
    )

    # Set padding_side to 'left' for decoder-only models
    tokenizer.padding_side = "left"

    # Set eos_token and pad_token explicitly
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    if tokenizer.pad_token is None:
        pad_token = "<pad>"
        if pad_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": pad_token})
        else:
            tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        cache_dir=hf_home,
        torch_dtype=torch.bfloat16,
    )

    # Update model's config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

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

def load_dataset(
    train_size: int,
    train_complex_file: str,
    train_simplified_file: str,
    test_complex_file: str,
    test_simplified_file: str
) -> Dict[str, Dataset]:
    """
    Load training and test datasets from specified files.

    Args:
        train_size (int): Number of samples to load for training.
        train_complex_file (str): Path to the training file containing complex sentences.
        train_simplified_file (str): Path to the training file containing simplified sentences.
        test_complex_file (str): Path to the test file containing complex sentences.
        test_simplified_file (str): Path to the test file containing simplified sentences.

    Returns:
        Dict[str, Dataset]: A dictionary with 'train' and 'test' datasets.
    """
    # Load training data
    with open(train_complex_file, "r") as f:
        complex_sentences = [line.strip() for line in f][:train_size]
    with open(train_simplified_file, "r") as f:
        simplified_sentences = [line.strip() for line in f][:train_size]

    train_inputs = [
        f"INSTRUKSJON: Del opp setningen under i atskilte setninger. \nINNDATA: {complex}\nRESPONS: "
        for complex in complex_sentences
    ]
    train_dataset = Dataset.from_dict({"inputs": train_inputs, "targets": simplified_sentences})

    # Load test data
    with open(test_complex_file, "r") as f:
        test_complex_sentences = [line.strip() for line in f] # Script works, no longer needed for dev [:train_size//50]
    with open(test_simplified_file, "r") as f:
        test_simplified_sentences = [line.strip() for line in f] # Script works, no longer needed for dev [:train_size//50]

    test_inputs = [
        f"INSTRUKSJON: Del opp setningen under i atskilte setninger. \nINNDATA: {complex}\nRESPONS: "
        for complex in test_complex_sentences
    ]
    test_dataset = Dataset.from_dict({"inputs": test_inputs, "targets": test_simplified_sentences})

    return {"train": train_dataset, "test": test_dataset}

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

def prepare_inference_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int = 320
) -> Dataset:
    def tokenize_function(examples):
        max_input_length = max_length - 2  # Reserve space for 2 eos_token_ids
        tokenized_inputs = tokenizer(
            examples["inputs"],
            padding=False,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )

        input_ids_batch = []
        for inp_ids in tokenized_inputs["input_ids"]:
            ids = inp_ids + [tokenizer.eos_token_id]  # Add single EOS for inference
            ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
            input_ids_batch.append(ids)

        return {"input_ids": input_ids_batch}

    return dataset.map(
        tokenize_function, batched=True, remove_columns=["inputs"]
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

        log_dir = os.getenv("LOG_DIR", "/scratch/project_465001453/mlworks-logs/")  # Fallback if unset
        tracking_uri = f"file:{log_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("NorwSynt")
        print(f"MLFLOW_TRACKING_URI set to: {tracking_uri}")

        datasets = load_dataset(
            train_complex_file="/workspace/nor/train_whole.txt",
            train_simplified_file="/workspace/nor/train_split.txt",
            test_complex_file="/workspace/nor/test_whole.txt",
            test_simplified_file="/workspace/nor/test_split.txt",
            train_size=args.train_size
        )

        train_dataset = datasets["train"]
        test_dataset = datasets["test"]

        # Split the dataset into train and dev sets
        train_dev_split = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
        train_dataset = train_dev_split["train"]
        dev_dataset = train_dev_split["test"]  # This is the dev set

        prepared_train_dataset = prepare_dataset(train_dataset, tokenizer)
        prepared_val_dataset = prepare_dataset(dev_dataset, tokenizer)

        # Generate a timestamp for readability
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate a UUID for uniqueness
        unique_id = uuid.uuid4().hex

        training_args = TrainingArguments(
            output_dir = f"/scratch/project_465001453/train/NorwSynt/N{timestamp}_{unique_id}",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to=["mlflow"],  # Changed to log training metrics to MLflow
            ddp_backend="nccl",
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=prepared_train_dataset,
            eval_dataset=prepared_val_dataset,
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer, padding=True, return_tensors="pt"
            ),
        )

        # Set MLflow experiment before training
        mlflow.start_run(run_name="training")
        mlflow.log_param("train_size", args.train_size)
        mlflow.log_param("model", args.model)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("per_device_train_batch_size", training_args.per_device_train_batch_size)

        trainer.train()

        # make sure to end the first mlflow run to avoid confusion
        mlflow.end_run()

        if dist.get_rank() == 0:
            logging.info("Training completed, waiting for filesystem sync...")
        time.sleep(10)  # Add delay for filesystem sync
        # Before proceeding to inference
        if dist.get_rank() == 0:
            logging.info("Verifying checkpoint directory permissions...")
            os.system("ls -l /scratch/project_465001453/train/NorwSynt/")

        if not args.skip_inference:
            # Get world size and rank
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # Split the test_inputs across all ranks
            test_inputs = test_dataset["inputs"]
            num_inputs = len(test_inputs)
            chunk_size = (num_inputs + world_size - 1) // world_size  # Ceiling division
            start = rank * chunk_size
            end = min(start + chunk_size, num_inputs)
            my_inputs = test_inputs[start:end]

            # Debug tokenizer settings
            if rank == 0:
                print(
                    f"Tokenizer pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}, padding_side: {tokenizer.padding_side}")

            # Perform inference on this rank's chunk
            my_responses = []
            model.eval()
            with torch.no_grad():
                for prompt in my_inputs:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        add_special_tokens=False,
                        return_attention_mask=True,  # Generate attention mask
                        padding=False,  # Minimal padding for sequential processing
                    ).to(model.device)

                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],  # Pass attention mask
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,  # Explicitly set
                    )

                    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    output_text = tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    my_responses.append(output_text)

            all_responses = [None] * world_size
            dist.gather_object(my_responses, all_responses if rank == 0 else None, dst=0)
            if rank == 0:
                all_responses = [item for sublist in all_responses for item in sublist]
                inferred_data = [{"prompt": prompt, "response": response} for prompt, response in
                                 zip(test_inputs, all_responses)]
                with open("inferred_sentences.json", "w") as f:
                    json.dump(inferred_data, f, indent=4)
                with mlflow.start_run(run_name="inference"):
                    mlflow.log_param("train_size", args.train_size)
                    mlflow.log_param("model", args.model)
                    mlflow.log_param("epochs", args.epochs)

                    mlflow.log_artifact("inferred_sentences.json")
                    print(
                        f"Inferred sentences saved as artifact 'inferred_sentences.json' in MLflow run. Train size: {args.train_size}, model: {args.model}")
                    references = [[ref] for ref in test_dataset["targets"]]
                    bert_score = load("bertscore")
                    sari_score = load("sari")
                    sacrebleu = load("sacrebleu")
                    bert_results = bert_score.compute(
                        predictions=all_responses, references=test_dataset["targets"], lang="no"
                    )
                    sari_results = sari_score.compute(
                        sources=[prompt.split("\nINNDATA: ")[1].split("\nRESPONS:")[0] for prompt in test_inputs],
                        predictions=all_responses,
                        references=references,
                    )
                    bleu_results = sacrebleu.compute(
                        predictions=all_responses, references=references
                    )
                    print("\nEvaluation Results:")
                    print(f"Training Size: {args.train_size}")
                    print(f"BERTScore Precision: {np.mean(bert_results['precision']):.4f}")
                    print(f"BERTScore Recall: {np.mean(bert_results['recall']):.4f}")
                    print(f"BERTScore F1: {np.mean(bert_results['f1']):.4f}")
                    print(f"SARI Score: {sari_results['sari']:.4f}")
                    print(f"SacreBLEU Score: {bleu_results['score']:.4f}")

                    # Logging to mlflow
                    mlflow.log_metric("test_bertscore_precision", np.mean(bert_results['precision']))
                    mlflow.log_metric("test_bertscore_recall", np.mean(bert_results['recall']))
                    mlflow.log_metric("test_bertscore_f1", np.mean(bert_results['f1']))
                    mlflow.log_metric("test_sari_score", sari_results['sari'])
                    mlflow.log_metric("test_sacrebleu_score", bleu_results['score'])


    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

def main():
    run_training_pipeline()

if __name__ == "__main__":
    main()