import argparse
import torch
import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NameReversalDataset(Dataset):
    def __init__(self, names, tokenizer, max_length=64):
        self.names = names
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        reversed_name = name[::-1]
        full_text = f"{self.tokenizer.bos_token}Reverse the name: {name}. Answer:<|sep|>{reversed_name}{self.tokenizer.eos_token}"
        prompt = f"{self.tokenizer.bos_token}Reverse the name: {name}. Answer:<|sep|>"

        return {
            'full_text': full_text,
            'prompt': prompt,
            'name': name,
            'reversed': reversed_name
        }


def generate_reversal(model, tokenizer, name, device, max_new_tokens=20, temperature=0.1):
    prompt = f"{tokenizer.bos_token}Reverse the name: {name}. Answer:<|sep|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the answer part
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[1].strip()
        # Remove any extra text after the first word (the reversed name)
        answer = answer.split()[0] if answer else ""
        return answer
    return ""


def evaluate_accuracy(model, tokenizer, test_names, device, temperature=0.1):
    correct = 0
    errors = []
    for name in tqdm(test_names, desc="Evaluating"):
        generated = generate_reversal(model, tokenizer, name, device, temperature=temperature)
        expected = name[::-1]
        if generated == expected:
            correct += 1
        else:
            errors.append((name, expected, generated))

    exact_match_acc = correct / len(test_names) if test_names else 0.0
    return {
        'exact_match_accuracy': exact_match_acc,
        'errors': errors
    }


def test_manual_examples(model, tokenizer, device, temperature=0.1):
    """Test on the manual examples from the notebook."""
    print("\nManual Examples:")
    print("="*50)

    # https://cmu-l3.github.io/anlp-fall2025/
    test_names = ['sean', 'joel', 'chen', 'dareen', 'neel', 'akshita', 'ashish', 'manan', 'sanidhya']

    correct = 0
    for name in test_names:
        generated = generate_reversal(model, tokenizer, name, device, temperature=temperature)
        expected = name[::-1]
        is_correct = generated == expected
        correct += is_correct
        symbol = "✓" if is_correct else "✗"
        print(f"{name:10} → {generated:10} (expected: {expected:10}) {symbol}")

    accuracy = correct / len(test_names) * 100
    print(f"\nManual Examples Accuracy: {correct}/{len(test_names)} = {accuracy:.1f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned model on name reversal task')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--data-path', type=str, default='names_split.json', help='Path to split data file (JSON with train/dev/test)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for data split')
    parser.add_argument('--show-errors', type=int, default=10, help='Number of errors to display')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--test-size', type=int, default=500, help='Number of test samples to evaluate on')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_name = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add special tokens
    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "bos_token": "<|startoftext|>",
        "sep_token": "<|sep|>",
    })
    tokenizer.padding_side = "left"

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data_splits = json.load(f)
    
    train_data = data_splits['train']
    dev_data = data_splits['dev']
    test_data = data_splits['test']
    if args.test_size > 0:
        test_data = test_data[:args.test_size]

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

    # Test on manual examples
    manual_accuracy = test_manual_examples(
        model, tokenizer, device,
        temperature=args.temperature
    )

    # Evaluate on test set
    print(f"\n{'='*50}")
    print(f"Evaluating on test set (sample size: {len(test_data)})...")
    print(f"{'='*50}")

    results = evaluate_accuracy(
        model, tokenizer, test_data, device, 
        temperature=args.temperature
    )

    test_accuracy = results['exact_match_accuracy'] * 100
    num_errors = len(results['errors'])
    num_evaluated = len(test_data)

    print(f"\nTest Set Exact Match Accuracy: {test_accuracy:.2f}%")
    print(f"Number of errors: {num_errors}/{num_evaluated}")

    # Show some errors
    if results['errors'] and args.show_errors > 0:
        print(f"\nFirst {args.show_errors} errors:")
        print(f"{'Name':<15} {'Expected':<15} {'Generated':<15}")
        print("="*50)
        for name, expected, generated in results['errors'][:args.show_errors]:
            print(f"{name:<15} {expected:<15} {generated:<15}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Manual Examples Accuracy: {manual_accuracy:.1f}%")
    print(f"Test Set Accuracy:        {test_accuracy:.2f}%")
    print(f"Checkpoint:               {args.checkpoint_path}")


if __name__ == '__main__':
    main()
