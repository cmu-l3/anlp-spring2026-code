import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add wandb logging
import wandb
wandb.init(project="name-reversal-rl", name="SmolLM2-135M-rl-finetune")


# --- load model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "sep_token": "<|sep|>",
})
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("model_sft.pt"))
tokenizer.padding_side = "left"
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- data loader
from torch.utils.data import Dataset, DataLoader
import random

class NameReversalDataset(Dataset):
    def __init__(self, names, tokenizer, max_length=64):
        self.names = names
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        prompt = f"{self.tokenizer.bos_token}Reverse the name: {name}. Answer:<|sep|>"
        return {
            'prompt': prompt,
            'name': name,
        }
    
    def collate_fn(self, batch):
        prompts = [item['prompt'] for item in batch]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }

# Split data
data = open('names.txt').read().splitlines()
random.seed(123)
random.shuffle(data)

n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))

train_data = data[:n1]
dev_data = data[n1:n2]
test_data = data[n2:]

print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

# Create datasets
train_dataset = NameReversalDataset(train_data, tokenizer)
dev_dataset = NameReversalDataset(dev_data, tokenizer)
test_dataset = NameReversalDataset(test_data, tokenizer)

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)


# -- reward function
def reward_function(output, prompt):
    # Extract the target name from the untokenized prompt
    prompt_str = tokenizer.decode(prompt, skip_special_tokens=True)
    target_name = prompt_str.split("Reverse the name:")[-1].split(".")[0].strip()
    # Decode the model output
    output_str = tokenizer.decode(output, skip_special_tokens=True)
    # Take the string after Answer: and prior to the first period.
    parsed_output = output_str.split("Answer:")[-1].split(".")[0].strip()
    if parsed_output == target_name[::-1]:
        return 1.0
    else:
        return 0.0


# -- RL training loop
import torch.optim as optim
from tqdm import tqdm

learning_rate = 5e-5
num_epochs = 3
K = 8  # number of samples per prompt
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
        
        # Generate outputs
        with torch.no_grad():
            ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 10,
                do_sample=True,
                num_return_sequences=K
            )
        
        optimizer.zero_grad()

        # Get log probabilities of generated tokens
        outputs = model(
            input_ids=ids,
            attention_mask=(ids != tokenizer.pad_token_id).long()
        )
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

        # Compute rewards
        rewards = []
        for i in range(ids.size(0)):
            reward = reward_function(ids[i], input_ids[i // K])
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Normalize rewards
        rewards_original = rewards.clone()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Get log probabilities of generated tokens (excluding prompt tokens)
        generated_ids = ids[:, input_ids.size(1):]
        log_probs = log_probs[:, input_ids.size(1)-1:-1, :]
        selected_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)

        # Mask loss after the first <|endoftext|> token
        eos_mask = (generated_ids == tokenizer.eos_token_id).float()
        eos_count = torch.cumsum(eos_mask, dim=1)
        loss_mask = (eos_count <= 1).float()
        selected_log_probs = selected_log_probs * loss_mask

        loss = - (rewards.unsqueeze(-1) * selected_log_probs)
        loss = loss.sum() / loss_mask.sum()
        # loss = - (rewards.unsqueeze(-1) * selected_log_probs).mean()
        # import ipdb; ipdb.set_trace()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % 1 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'reward': f'{rewards_original.mean().item():.4f}'})
            # Log to wandb
            wandb.log({
                'train/loss': avg_loss,
                'train/reward': rewards_original.mean().item(),
                'train/lr': scheduler.get_last_lr()[0],
            })
        
        # Periodically save the checkpoint
        if (batch_idx + 1) % 500 == 0:
            torch.save(model.state_dict(), f"model_rl_checkpoint_epoch_{epoch+1}_step_{batch_idx+1}.pt")
    
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