import csv
import unicodedata
import wave
from pathlib import Path

import jiwer
import numpy as np
import sherpa_onnx

# run_offline.py nằm trong src/, nên đi lên một cấp để tìm dữ liệu.
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
SAMPLE_RATE = 16000


def normalize(text: str) -> str:
    """Chuẩn hóa chữ hoa, dấu câu và khoảng trắng trước khi chấm."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return " ".join(text.split())


def read_wav(path: Path) -> np.ndarray:
    """Đọc WAV mono 16 kHz, PCM 16-bit thành mẫu float32."""
    with wave.open(str(path), "rb") as wav:
        audio_format = (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
        )
        if audio_format != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"Cần WAV mono 16 kHz, PCM 16-bit: {path}")

        pcm = wav.readframes(wav.getnframes())

    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def main() -> None:
    with (ROOT / "references.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        clips = list(csv.DictReader(file))

    expected_ids = [f"utt_{index:02d}" for index in range(1, 13)]
    if [clip["id"] for clip in clips] != expected_ids:
        raise ValueError("references.csv phải có đủ utt_01 đến utt_12 theo thứ tự.")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL / "encoder.int8.onnx"),
        decoder=str(MODEL / "decoder.onnx"),
        joiner=str(MODEL / "joiner.int8.onnx"),
        tokens=str(MODEL / "tokens.txt"),
        provider="cpu",
        num_threads=1,
        sample_rate=SAMPLE_RATE,
        decoding_method="greedy_search",
    )

    streams = []
    for clip in clips:
        samples = read_wav(ROOT / clip["path"])
        stream = recognizer.create_stream()

        # Offline: nạp toàn bộ mẫu của một clip trong một lần.
        stream.accept_waveform(SAMPLE_RATE, samples)
        streams.append(stream)

    recognizer.decode_streams(streams)
    transcripts = [stream.result.text for stream in streams]

    output_path = ROOT / "predictions_offline.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "configuration", "path", "reference", "hypothesis"])

        for clip, transcript in zip(clips, transcripts, strict=True):
            writer.writerow(
                [
                    clip["id"],
                    "zipformer_offline",
                    clip["path"],
                    clip["reference"],
                    transcript,
                ]
            )
            print(f"{clip['id']}: {transcript}")

    references = [normalize(clip["reference"]) for clip in clips]
    hypotheses = [normalize(transcript) for transcript in transcripts]

    word_scores = jiwer.process_words(references, hypotheses)
    char_score = jiwer.cer(
        [text.replace(" ", "") for text in references],
        [text.replace(" ", "") for text in hypotheses],
    )

    print(f"\nWER toàn bộ: {word_scores.wer:.2%}")
    print(f"CER toàn bộ: {char_score:.2%}")
    print(
        f"Lỗi thay: {word_scores.substitutions}, "
        f"lỗi bỏ: {word_scores.deletions}, lỗi chèn: {word_scores.insertions}"
    )
    print(f"Đã lưu {len(transcripts)} transcript vào {output_path.name}")


if __name__ == "__main__":
    main()