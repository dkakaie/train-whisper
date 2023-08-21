import torch
import torch.nn as nn
import whisper

from dataset import Collator, CustomLibriSpeechDataset, IGNORE_TOKEN
from dims import dims
from utils import calculate_wer, LinearLearningRateDecayWithWarmup, train, validate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.0015
BATCH_SIZE = 16
EPOCHS = 15

train_dataset = CustomLibriSpeechDataset(split="train-clean-360", device=DEVICE)
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

loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1
)

WARMUP_STEPS = 2048
TOTAL_STEPS = len(train_data_loader) * EPOCHS
assert WARMUP_STEPS < TOTAL_STEPS

lr_lambda = LinearLearningRateDecayWithWarmup(warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


for i in range(EPOCHS):
    train(model, train_data_loader, loss_fn, optimizer, scheduler)
    validate(model, val_data_loader, loss_fn)
    val_wer = calculate_wer(model, val_data_loader)
    print(f"Epoch {i + 1} WER: {val_wer}")


