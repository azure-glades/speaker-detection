"""
SpeakerDiarizer
A thin, reusable wrapper around pyannote.audio 3.1
------------------------------------------------------------------
Usage
-----
from speaker_diarizer import SpeakerDiarizer
dia = SpeakerDiarizer(hf_token="YOUR_HUGGINGFACE_TOKEN")
segments = dia(waveform, sample_rate=16_000)
# segments -> [(start, end, speaker_label), ...]
"""

import torch
from pyannote.audio import Pipeline
import numpy as np
from typing import List, Tuple


class SpeakerDiarizer:
    """
    1. Loads the pyannote speaker-diarization-3.1 model once
    2. Accepts mono or stereo numpy arrays (float32 [-1, 1] or int16)
    3. Returns list of (start, end, speaker) tuples in seconds
    """

    def __init__(
        self,
        hf_token: str,
        device: str = None,
        min_segment_duration: float = 0.2,
    ):
        """
        Parameters
        ----------
        hf_token : str
            HuggingFace access token with permissions for pyannote models.
        device : str, optional
            'cpu', 'cuda', 'cuda:0', etc.  Auto-detected if None.
        min_segment_duration : float
            Discard segments shorter than this (seconds) to reduce chatter.
        """
        self.min_segment_duration = min_segment_duration
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float, str]]:
        """
        Run diarization.

        Parameters
        ----------
        waveform : np.ndarray
            Shape (samples,) for mono or (channels, samples) for multi-channel.
            Values float32 in [-1, 1] or int16 in [-32768, 32767].
        sample_rate : int
            Sample rate of `waveform`.

        Returns
        -------
        segments : list[tuple]
            (start_time, end_time, speaker_label) in seconds.
        """
        # --- 1. Convert to mono float32 tensor ------------------------ #
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)  # simple mono mix
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        # --- 2. Run pyannote pipeline -------------------------------- #
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )

        # --- 3. Flatten to list and filter short segments ------------ #
        segments = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
            if turn.end - turn.start >= self.min_segment_duration
        ]
        return segments