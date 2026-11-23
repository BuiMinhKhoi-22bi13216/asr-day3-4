# Thực hành ASR — Đợt 2, Ngày 3–4

## Dữ liệu và môi trường

- 12 clip đọc tiếng Việt từ 12 người đọc trong Common Voice 27.0, tập test.
- WAV mono 16 kHz, PCM 16-bit, lưu trong `clips/`.
- `references.csv` chứa mã clip, đường dẫn và câu chuẩn.
- Môi trường chạy là Linux/WSL, Python 3.14, CPU, một luồng.

## Cách chạy lại

Mở terminal WSL tại thư mục chứa `references.csv`. Kích hoạt môi trường
Python rồi cài các thư viện:

```bash
python -m pip install sherpa-onnx numpy jiwer pytest ruff
python -m pip install --only-binary=:all: faster-whisper
```

Chuẩn bị model tại thư mục gốc:

- [Zipformer offline INT8](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2) — giải nén thành `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/`.
- [Zipformer streaming FP16](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h/tree/main) — lấy ba file ONNX `chunk-32-left-128`, lưu lần lượt thành `encoder.onnx`, `decoder.onnx`, `joiner.onnx` trong `zipformer-streaming/`. Lưu `config.json` của model thành `tokens.txt` trong cùng thư mục.
- Whisper `small` được tải tự động khi chạy lần đầu.

```bash
python -m src.run_offline
python -m src.run_streaming
python -m src.run_whisper
python -m pytest -q
python -m ruff check src tests
```

## Kết quả tự đo

| Cấu hình | WER | CER | Thay / bỏ / chèn |
|---|---:|---:|---:|
| Zipformer offline, greedy | 4,41% | 2,70% | 4 / 1 / 1 |
| Zipformer streaming, greedy | 5,88% | 5,17% | 6 / 2 / 0 |
| Whisper small, CPU INT8, beam size 1 | 22,06% | 12,36% | 27 / 2 / 1 |

Cả ba lượt dùng cùng câu chuẩn và quy tắc chấm. WER tách theo khoảng
trắng sau khi chuẩn hóa chữ thường, Unicode và dấu câu. CER bỏ thêm
khoảng trắng. Ví dụ, Whisper có 30 lỗi trên 136 đơn vị, cho WER 22,06%.

## Quan sát từ kết quả

- `utt_01`: Streaming trả `EM` sau khi nhận 1,50 giây audio, rồi trả
  `EM ĐÃ CHƯA KỊP ĐÁP` ở mốc 2,80 giây. Câu chuẩn bắt đầu bằng `Anh ta`.
  Xem `streaming_utt_01.txt`. Các mốc là lượng audio đã cấp vào.
- `utt_05`: Hai bản Zipformer nghe `giỗ đầu` thành `dỗ đầu`. Whisper
  nhận đúng cụm này.
- `utt_12`: Zipformer offline giữ đúng `Word Excel`, còn Whisper giữ
  đúng `Firefox Chrome`. Cả hai đều sai một phần danh sách tên phần mềm.

Zipformer offline có WER và CER thấp nhất trên bộ thử này. Bộ dữ liệu
gồm các câu đọc ngắn, nên kết quả phản ánh ba cấu hình trên đúng 12 clip
đã chọn. Các tình huống hội thoại dài, nói chồng lời và mic xa nằm ngoài
phạm vi thử nghiệm.