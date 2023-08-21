from typing import Optional

from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
import jiwer
import torch
import torch.nn as nn
import whisper


class LinearLearningRateDecayWithWarmup:
    def __init__(self, warmup_steps: int, total_steps: int):
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

    def __call__(self, current_step: int) -> float:
        # linear increase to 1.0
        if current_step < self._warmup_steps:
            return current_step / self._warmup_steps
        # linear decrease to 0.0
        return max(
            (self._total_steps - current_step) / (self._total_steps - self._warmup_steps),
            0.0
        )


def train(
    model: whisper.model.Whisper,
    train_data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    print_interval: int = 100
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

        if i % print_interval == 0:
            print(f"loss: {loss.item():>7f}  [{i}/ {n_batches}]")


def validate(
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
            whisper.DecodingOptions(language="en", without_timestamps=True)
        )
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    if normalize:
        normalizer = EnglishTextNormalizer()
        hypotheses = [normalizer(s) for s in hypotheses]
        references = [normalizer(s) for s in references]

    return jiwer.wer(hypothesis=hypotheses, reference=references) * 100