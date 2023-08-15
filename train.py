from tqdm import tqdm
import jiwer
import torch
import torch.nn as nn
import whisper

from dataset import Collator, CustomLibriSpeechDataset, IGNORE_TOKEN
from dims import dims
from utils import LinearLearningRateDecayWithWarmup
from whisper.normalizers import EnglishTextNormalizer


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


def train(
    model: whisper.model.Whisper,
    train_data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
) -> None:
    n_batches = len(train_data_loader)
    model.train()
    for i, (mels, input_tokens, target_tokens, _) in enumerate(train_data_loader):
        optimizer.zero_grad()
        output = model(mels, input_tokens)
        b, t, c = output.shape
        loss = loss_fn(output.view(b * t, c), target_tokens.view(b * t))

        # back propagation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # step learning rate scheduler
        scheduler.step()

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


def calculate_wer(
    model: whisper.model.Whisper,
    data_loader: torch.utils.data.DataLoader,
    normalize: bool = True,
) -> float:
    hypotheses, references = [], []
    for mels, _, _, texts in tqdm(data_loader):
        results = model.decode(
            mels,
            whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
        )
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    if normalize:
        normalizer = EnglishTextNormalizer()
        hypotheses = [normalizer(s) for s in hypotheses]
        references = [normalizer(s) for s in references]

    wer = jiwer.wer(hypothesis=hypotheses, reference=references) * 100
    return wer


for i in range(EPOCHS):
    train(model, train_data_loader, loss_fn, optimizer)
    validation(model, val_data_loader, loss_fn)
    val_wer = calculate_wer(model, val_data_loader)
    print(f"Epoch {i + 1} WER: {val_wer}")


