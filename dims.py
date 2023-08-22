from whisper.model import ModelDimensions


dims = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=1024,
    n_audio_head=16,
    n_audio_layer=24,
    n_vocab=51864,
    n_text_ctx=448,
    n_text_state=1024,
    n_text_head=16,
    n_text_layer=24,
)