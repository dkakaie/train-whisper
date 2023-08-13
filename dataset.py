from typing import List, Tuple

from whisper.tokenizer import get_tokenizer
import torch
import torch.nn.functional as F
import torchaudio
import whisper


IGNORE_TOKEN = -1


class CustomLibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "test-clean", device: str = "cpu", ignore_token: int = IGNORE_TOKEN):
        self._dataset = torchaudio.datasets.LIBRISPEECH(
            root="./data/", url=split, download=True,
        )
        self._device = device
        self._ignore_token = ignore_token

        self._tokenizer = get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )
        self._sot_sequence = self._tokenizer.sot_sequence_including_notimestamps
        self._eot = self._tokenizer.eot

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        audio, _, text, _, _, _ = self._dataset[idx]

        audio = whisper.pad_or_trim(audio.flatten()).to(self._device)
        mel = whisper.log_mel_spectrogram(audio)

        tokens = torch.tensor(self._tokenizer.encode(text))
        input_tokens = torch.concatenate([
            torch.tensor(self._sot_sequence), tokens
        ]).to(self._device)
        target_tokens = torch.concatenate([
            torch.tensor([self._ignore_token]), tokens, torch.tensor([self._eot])
        ]).to(self._device)

        return mel, input_tokens, target_tokens, text


class Collator:
    def __init__(self, ignore_token: int = IGNORE_TOKEN):
        self._eot = get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        ).eot
        self._ignore_token = ignore_token

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        max_token_length = self._get_batch_max_token_length(batch)

        mel_lst, input_tokens_lst, target_tokens_lst, text_lst = [], [], [], []
        for mel, input_tokens, target_tokens, text in batch:
            mel_lst.append(mel)
            text_lst.append(text)
            padding = max_token_length - input_tokens.shape[0]
            if padding > 0:
                input_tokens_lst.append(F.pad(input_tokens, (0, padding), value=self._eot))
                target_tokens_lst.append(F.pad(target_tokens, (0, padding), value=self._ignore_token))
            else:
                input_tokens_lst.append(input_tokens)
                target_tokens_lst.append(target_tokens)
        return torch.stack(mel_lst), torch.stack(input_tokens_lst), torch.stack(target_tokens_lst), text_lst

    @staticmethod
    def _get_batch_max_token_length(batch) -> int:
        max_length = 0
        for _, input_tokens, _, _ in batch:
            length = input_tokens.shape[0]
            if length > max_length:
                max_length = length

        return max_length
