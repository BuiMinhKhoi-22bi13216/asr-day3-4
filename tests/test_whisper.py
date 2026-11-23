from types import SimpleNamespace
from unittest.mock import Mock, sentinel

from src.run_whisper import transcribe_clip


def test_transcribe_clip_joins_segments_and_uses_vietnamese():
    model = Mock()
    segments = iter(
        [
            SimpleNamespace(text=" Xin chào "),
            SimpleNamespace(text=" mọi người. "),
        ]
    )
    model.transcribe.return_value = (segments, None)

    transcript = transcribe_clip(model, sentinel.samples)

    assert transcript == "Xin chào mọi người."
    model.transcribe.assert_called_once_with(
        sentinel.samples,
        language="vi",
        beam_size=1,
        vad_filter=False,
    )