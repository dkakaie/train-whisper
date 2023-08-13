import torch
import torch.nn as nn
import whisper

from dataset import Collator, CustomLibriSpeechDataset, IGNORE_TOKEN
from dims import dims
from utils import calculate_wer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL = 500
LR = 1e-5
BATCH_SIZE = 16
EPOCHS = 5

train_dataset = CustomLibriSpeechDataset(split="dev-clean", device=DEVICE)
val_dataset = CustomLibriSpeechDataset(split="dev-clean", device=DEVICE)

collator = Collator()
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

model = whisper.model.Whisper(dims)

# todo: add warming up phase
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def take_train_step(
    model: whisper.model.Whisper,
    mels: torch.Tensor,
    input_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    loss_fn,
    optimizer,
) -> float:
    model.train()
    output = model(mels, input_tokens)
    b, t, c = output.shape

    loss = loss_fn(output.view(b * t, c), target_tokens.view(b * t))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


for i, batch in enumerate(train_data_loader):
    mels, input_tokens, target_tokens, _ = batch
    train_step_loss_val = take_train_step(model, mels, input_tokens, target_tokens, loss_fn, optimizer)
    print(f"Batch no: {i}, train step loss: {train_step_loss_val}")



