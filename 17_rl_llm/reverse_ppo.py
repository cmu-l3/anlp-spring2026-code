import torch
import json
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Settings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_split_path', type=str, default='names_split.json')
parser.add_argument('--model_name', type=str, default="HuggingFaceTB/SmolLM2-135M")
parser.add_argument('--sft_model_path', type=str, default='model_sft.pt')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--num_samples_per_prompt', type=int, default=8)
parser.add_argument('--ppo_epochs', type=int, default=3)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--save_path', type=str, default='model_rl.pt')
parser.add_argument('--wandb_project', type=str, default='name-reversal-rl')
parser.add_argument('--wandb_run_name', type=str, default='SmolLM2-135M-rl-finetune')
args = parser.parse_args()

data_split_path = args.data_split_path
model_name = args.model_name
sft_model_path = args.sft_model_path
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_samples_per_prompt = args.num_samples_per_prompt
ppo_epochs = args.ppo_epochs
ppo_epsilon_low = 0.8
ppo_epsilon_high = 2.0
grad_clip = args.grad_clip
save_interval = args.save_interval
save_overwrite = True
save_path = args.save_path
wandb_project = args.wandb_project
wandb_run_name = args.wandb_run_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- wandb logging
wandb.init(project=wandb_project, name=wandb_run_name)

# --- load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "sep_token": "<|sep|>",
})
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(sft_model_path))
tokenizer.padding_side = "left"
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- data loader
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

# Load data
with open(data_split_path, 'r') as f:
    data_splits = json.load(f)

train_data = data_splits['train']
dev_data = data_splits['dev']
test_data = data_splits['test']
print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

# Create datasets
train_dataset = NameReversalDataset(train_data, tokenizer)
dev_dataset = NameReversalDataset(dev_data, tokenizer)
test_dataset = NameReversalDataset(test_data, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)


# -- reward function
def reward_function(output, prompt):
    # Extract the target name from the untokenized prompt
    prompt_str = tokenizer.decode(prompt, skip_special_tokens=True)
    target_name = prompt_str.split("Reverse the name:")[-1].split(".")[0].strip()
    # Decode the model output, take the string after Answer: and prior to the first period.
    output_str = tokenizer.decode(output, skip_special_tokens=True)
    parsed_output = output_str.split("Answer:")[-1].split(".")[0].strip()
    if parsed_output == target_name[::-1]:
        return 1.0
    else:
        return 0.0


# -- RL training loop
model = model.to(device)

# Old model for PPO-style loss computations
old_model = AutoModelForCausalLM.from_pretrained(model_name)
old_model.resize_token_embeddings(len(tokenizer))
old_model = old_model.to(device)
old_model.load_state_dict(model.state_dict())
old_model.eval()

# Optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    avg_reward_log = 0
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # -- Generate outputs
        with torch.no_grad():
            ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 10,
                do_sample=True,
                num_return_sequences=num_samples_per_prompt
            )
        
        # -- Compute rewards and advantages
        rewards = []
        for i in range(ids.size(0)):
            reward = reward_function(ids[i], input_ids[i // num_samples_per_prompt])
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards_original = rewards.clone()

        # Get the mean per prompt
        means = rewards.view(-1, num_samples_per_prompt).mean(dim=1).repeat_interleave(num_samples_per_prompt)
        advantages = rewards - means

        # Normalize by batch statistics
        #  https://arxiv.org/pdf/2506.10910, https://arxiv.org/pdf/2006.05990
        mean = advantages.mean()
        std = advantages.std()
        if std > 0:
            advantages = (advantages - mean) / (std + 1e-8)

        # -- Compute old log probabilities
        generated_ids = ids[:, input_ids.size(1):]
        with torch.no_grad():
            old_outputs = old_model(
                input_ids=ids,
                attention_mask=(ids != tokenizer.pad_token_id).long()
            )
            old_log_probs = torch.log_softmax(old_outputs.logits, dim=-1)
            old_selected = old_log_probs[:, input_ids.size(1)-1:-1, :].gather(
                2, generated_ids.unsqueeze(-1)).squeeze(-1)

        # -- PPO-style loss and updates
        for ppo_epoch in range(ppo_epochs):
            total_loss = 0
            optimizer.zero_grad()

            # Get log probabilities of generated tokens (excluding prompt tokens)
            outputs = model(
                input_ids=ids,
                attention_mask=(ids != tokenizer.pad_token_id).long()
            )
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            log_probs = log_probs[:, input_ids.size(1)-1:-1, :]
            selected_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)

            #  PPO clipped objective
            ratio = torch.exp(selected_log_probs - old_selected)
            clipped_ratio = torch.clamp(ratio, ppo_epsilon_low, ppo_epsilon_high)
            loss = -torch.min(ratio * advantages.unsqueeze(-1), clipped_ratio * advantages.unsqueeze(-1))

            # Mask loss after the first <|endoftext|> token
            eos_mask = (generated_ids == tokenizer.eos_token_id).float()
            eos_count = torch.cumsum(eos_mask, dim=1)
            loss_mask = (eos_count <= 1).float()
            loss = loss * loss_mask
            
            # Sum over tokens, mean over batch
            if loss_mask.sum() > 0:
                loss = loss.sum(dim=1) / loss_mask.sum(dim=1)
            loss = loss.mean()

            # Backpropagation and optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Update old model with new weights
        old_model.load_state_dict(model.state_dict())

        # Update progress bar
        if batch_idx == 0:
            avg_reward_log = rewards_original.mean().item()
        else:
          avg_reward_log = 0.9 * avg_reward_log + 0.1 * rewards_original.mean().item()
        if batch_idx % 1 == 0:
            avg_loss = total_loss / (ppo_epochs)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 
                                      'reward': f'{rewards_original.mean().item():.4f}',
                                      'avg_reward': f'{avg_reward_log:.4f}'})
            wandb.log({
                'train/loss': avg_loss,
                'train/reward': rewards_original.mean().item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/avg_reward': avg_reward_log,
            })
        
        # Periodically save the checkpoint
        if (batch_idx + 1) % save_interval == 0:
            if save_overwrite:
                torch.save(model.state_dict(), save_path)
            else:
                filename = f"{epoch+1}_step_{batch_idx+1}_"+save_path
                torch.save(model.state_dict(), filename)
            # Save information about the latest checkpoint (settings, step, epoch)
            checkpoint_info = {
                'epoch': epoch + 1,
                'step': batch_idx + 1,
                'model_name': model_name,
                'sft_model_path': sft_model_path,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'num_samples_per_prompt': num_samples_per_prompt,
                'ppo_epochs': ppo_epochs,
                'grad_clip': grad_clip,
            }
            with open('latest_checkpoint_info.json', 'w') as f:
                json.dump(checkpoint_info, f)

    print(f"Epoch {epoch+1}/{num_epochs}")
