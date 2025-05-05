import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Inference Script")
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to the test data file (prompts, one per line)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=320, help="Maximum sequence length")
    parser.add_argument("--output_file", default="inferred_sentences.json", help="Path to save output JSON")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set padding_side to 'left'
    tokenizer.padding_side = 'left'

    # Ensure distinct pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        pad_token = "<pad>"
        if pad_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": pad_token})
            model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Read test prompts
    with open(args.test_data, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Define collate_fn
    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return encodings

    # Create DataLoader
    dataloader = DataLoader(
        prompts,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Perform inference
    responses = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                do_sample=True,
                temperature=0.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i in range(outputs.shape[0]):
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[i, input_len:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(output_text)

    # Create inferred_data
    inferred_data = [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, responses)]

    # Save to output_file
    with open(args.output_file, "w") as f:
        json.dump(inferred_data, f, indent=4)

    print(f"Inferred sentences saved to {args.output_file}")

if __name__ == "__main__":
    main()