from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
import jiwer
import torch
import whisper


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

    wer = jiwer.wer(hypothesis=hypotheses, reference=references)
    return wer
