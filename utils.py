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