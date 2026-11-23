from unittest.mock import Mock

import numpy as np
from src.run_streaming import FEED_SAMPLES, TAIL_SAMPLES, decode_clip


def test_decode_clip_feeds_chunks_and_finishes():
    recognizer = Mock()
    recognizer.is_ready.return_value = False
    recognizer.get_result.side_effect = ["xin", "xin", "xin chào"]
    stream = recognizer.create_stream.return_value
    samples = np.zeros(FEED_SAMPLES + 10, dtype=np.float32)

    final, partials = decode_clip(recognizer, samples)

    lengths = [len(call.args[1]) for call in stream.accept_waveform.call_args_list]
    assert lengths == [FEED_SAMPLES, 10, TAIL_SAMPLES]
    stream.input_finished.assert_called_once_with()
    assert partials == ["partial 0.10s | xin"]
    assert final == "xin chào"