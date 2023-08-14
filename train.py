import torch
import torch.nn as nn
import whisper

from dataset import Collator, CustomLibriSpeechDataset, IGNORE_TOKEN
from dims import dims
from utils import calculate_wer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL = 500
LR = 1e-5
BATCH_SIZE = 32
EPOCHS = 5

train_dataset = CustomLibriSpeechDataset(split="train-other-500", device=DEVICE)
val_dataset = CustomLibriSpeechDataset(split="test-clean", device=DEVICE)

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

model = whisper.model.Whisper(dims).to(DEVICE)

# todo: add warming up phase
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def train(
    model: whisper.model.Whisper,
    train_data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
) -> None:
    n_batches = len(train_data_loader)
    model.train()
    for i, (mels, input_tokens, target_tokens, _) in enumerate(train_data_loader):
        output = model(mels, input_tokens)
        b, t, c = output.shape
        loss = loss_fn(output.view(b * t, c), target_tokens.view(b * t))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{i}/ {n_batches}]")


def validation(
    model: whisper.model.Whisper,
    val_data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss.CrossEntropyLoss,
) -> None:
    n_batches = len(val_data_loader)
    total_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (mels, input_tokens, target_tokens, _) in enumerate(val_data_loader):
            output = model(mels, input_tokens)
            b, t, c = output.shape
            loss = loss_fn(output.view(b * t, c), target_tokens.view(b * t))
            total_val_loss += loss.item()

    print(f"Validation loss: {total_val_loss / n_batches}")


for _ in range(EPOCHS):
    train(model, train_data_loader, loss_fn, optimizer)
    validation(model, val_data_loader, loss_fn)
    val_wer = calculate_wer(model, val_data_loader)
    print(f"WER: {val_wer}")


