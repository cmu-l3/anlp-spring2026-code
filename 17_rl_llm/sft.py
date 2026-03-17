import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import random
import json
import torch.optim as optim
from tqdm import tqdm

# --- settings
data_filename = 'names.txt'
data_save_path = 'names_split.json'
model_name = "HuggingFaceTB/SmolLM2-135M"
model_save_path = 'model_sft.pt'
batch_size = 16
learning_rate = 5e-5
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add special tokens
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "sep_token": "<|sep|>",
})
model.resize_token_embeddings(len(tokenizer))

# Pad left
tokenizer.padding_side = "left"
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# --- data
data = open(data_filename).read().splitlines()
print(f"Total names: {len(data)}")

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
    
    def collate_fn(self, batch):
        full_texts = [item['full_text'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        
        tokenized = self.tokenizer(
            full_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = tokenized['input_ids'].clone()
        
        # Mask the prompt tokens in the labels (set to -100 so they're ignored in loss)
        for i, prompt in enumerate(prompts):
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_len = len(prompt_tokens)
            labels[i, :prompt_len] = -100  # Ignore prompt tokens in loss calculation
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }

# Split data
random.seed(123)
random.shuffle(data)

n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))

train_data = data[:n1]
dev_data = data[n1:n2]
test_data = data[n2:]

# Save the splits so that we can use them later
with open(data_save_path, 'w') as f:
    json.dump({
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }, f, indent=4)

print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

# Create datasets
train_dataset = NameReversalDataset(train_data, tokenizer)
dev_dataset = NameReversalDataset(dev_data, tokenizer)
test_dataset = NameReversalDataset(test_data, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

# Look at a sample batch
sample_batch = next(iter(train_loader))
print("Sample batch shape:")
print(f"  input_ids: {sample_batch['input_ids'].shape}")
print(f"  attention_mask: {sample_batch['attention_mask'].shape}")
print(f"  labels: {sample_batch['labels'].shape}")

# Decode first example
print("\nFirst example:")
input_text = tokenizer.decode(sample_batch['input_ids'][0], skip_special_tokens=False)
print(f"Input: {input_text}")

# Show which tokens are masked in labels
label_tokens = sample_batch['labels'][0]
valid_label_indices = (label_tokens != -100).nonzero(as_tuple=True)[0]
if len(valid_label_indices) > 0:
    valid_labels = label_tokens[valid_label_indices]
    label_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
    print(f"Target (answer only): {label_text}")


# --- training loop
import torch.optim as optim
from tqdm import tqdm

learning_rate = 5e-5
num_epochs = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
total_steps = len(train_loader) * num_epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_val_loss += outputs.loss.item()
    
    avg_val_loss = total_val_loss / len(dev_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Average training loss: {avg_train_loss:.4f}")
    print(f"  Average validation loss: {avg_val_loss:.4f}")

# Save
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

print("SFT done. To evaluate use 'python evaluate.py <checkpoint_path>'")