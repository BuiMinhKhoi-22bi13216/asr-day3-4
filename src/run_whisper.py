import csv

import jiwer
from faster_whisper import WhisperModel

from src.run_offline import ROOT, normalize, read_wav


def transcribe_clip(model: WhisperModel, samples) -> str:
    """Nhận dạng một clip và ghép các segment thành một câu."""
    segments, _ = model.transcribe(
        samples,
        language="vi",
        beam_size=1,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments)


def main() -> None:
    with (ROOT / "references.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        clips = list(csv.DictReader(file))

    expected_ids = [f"utt_{index:02d}" for index in range(1, 13)]
    if [clip["id"] for clip in clips] != expected_ids:
        raise ValueError("references.csv phải có đủ utt_01 đến utt_12 theo thứ tự.")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
    )

    transcripts = []
    for clip in clips:
        samples = read_wav(ROOT / clip["path"])
        transcript = transcribe_clip(model, samples)
        transcripts.append(transcript)
        print(f"{clip['id']}: {transcript}")

    output = ROOT / "predictions_whisper.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "configuration", "path", "reference", "hypothesis"])

        for clip, transcript in zip(clips, transcripts, strict=True):
            writer.writerow(
                [
                    clip["id"],
                    "whisper_small",
                    clip["path"],
                    clip["reference"],
                    transcript,
                ]
            )

    references = [normalize(clip["reference"]) for clip in clips]
    hypotheses = [normalize(transcript) for transcript in transcripts]

    words = jiwer.process_words(references, hypotheses)
    cer = jiwer.cer(
        [text.replace(" ", "") for text in references],
        [text.replace(" ", "") for text in hypotheses],
    )

    print(f"\nWER Whisper: {words.wer:.2%}")
    print(f"CER Whisper: {cer:.2%}")
    print(
        f"Lỗi thay: {words.substitutions}, "
        f"lỗi bỏ: {words.deletions}, lỗi chèn: {words.insertions}"
    )
    print(f"Đã lưu {len(transcripts)} transcript vào {output.name}")


if __name__ == "__main__":
    main()