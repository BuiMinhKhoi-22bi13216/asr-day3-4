import csv

import jiwer
import numpy as np
import sherpa_onnx

from src.run_offline import ROOT, SAMPLE_RATE, normalize, read_wav

MODEL = ROOT / "zipformer-streaming"
FEED_SAMPLES = SAMPLE_RATE // 10  # 100 ms ở 16 kHz
TAIL_SAMPLES = int(0.66 * SAMPLE_RATE)
DEMO_ID = "utt_01"


def decode_clip(recognizer, samples: np.ndarray) -> tuple[str, list[str]]:
    stream = recognizer.create_stream()
    partials = []
    previous = ""

    for start in range(0, len(samples), FEED_SAMPLES):
        chunk = samples[start : start + FEED_SAMPLES]
        stream.accept_waveform(SAMPLE_RATE, chunk)

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        current = recognizer.get_result(stream)
        if current and current != previous:
            fed_seconds = (start + len(chunk)) / SAMPLE_RATE
            partials.append(f"partial {fed_seconds:.2f}s | {current}")
            previous = current

    # Thêm đệm để giải mã các frame còn lại ở cuối câu.
    stream.accept_waveform(SAMPLE_RATE, np.zeros(TAIL_SAMPLES, dtype=np.float32))
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    return recognizer.get_result(stream), partials


def main() -> None:
    with (ROOT / "references.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        clips = list(csv.DictReader(file))

    expected_ids = [f"utt_{index:02d}" for index in range(1, 13)]
    if [clip["id"] for clip in clips] != expected_ids:
        raise ValueError("references.csv phải có đủ utt_01 đến utt_12 theo thứ tự.")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(MODEL / "encoder.onnx"),
        decoder=str(MODEL / "decoder.onnx"),
        joiner=str(MODEL / "joiner.onnx"),
        tokens=str(MODEL / "tokens.txt"),
        provider="cpu",
        num_threads=1,
        sample_rate=SAMPLE_RATE,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )

    rows = []
    for clip in clips:
        samples = read_wav(ROOT / clip["path"])
        transcript, partials = decode_clip(recognizer, samples)
        rows.append([clip["id"], "zipformer_streaming", clip["path"],
                     clip["reference"], transcript])
        print(f"{clip['id']}: {transcript}")

        if clip["id"] == DEMO_ID:
            log_lines = [f"Clip {DEMO_ID}", *partials, f"final | {transcript}"]
            (ROOT / "streaming_utt_01.txt").write_text(
                "\n".join(log_lines) + "\n", encoding="utf-8"
            )

    output = ROOT / "predictions_streaming.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "configuration", "path", "reference", "hypothesis"])
        writer.writerows(rows)

    references = [normalize(clip["reference"]) for clip in clips]
    hypotheses = [normalize(row[4]) for row in rows]
    word_scores = jiwer.process_words(references, hypotheses)
    char_score = jiwer.cer(
        [text.replace(" ", "") for text in references],
        [text.replace(" ", "") for text in hypotheses],
    )

    print(f"\nWER streaming: {word_scores.wer:.2%}")
    print(f"CER streaming: {char_score:.2%}")
    print(
        f"Lỗi thay: {word_scores.substitutions}, "
        f"lỗi bỏ: {word_scores.deletions}, lỗi chèn: {word_scores.insertions}"
    )
    print(f"Đã lưu {len(rows)} transcript vào {output.name}")


if __name__ == "__main__":
    main()