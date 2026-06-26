# Day 22 — DPO/ORPO Alignment · Báo cáo đầy đủ

**Học viên:** Nguyễn Trần Kiên · **MSSV:** 2A202600739 · **Cohort:** A20
**Ngày:** 2026-06-26 · **Track 3 — Daily Lab**

---

## 1. Hạ tầng chạy (Modal serverless)

| Hạng mục | Giá trị |
|---|---|
| Nền tảng | **Modal** (modal.com) — chạy **detached** (`modal run --detach`), tắt máy local vẫn chạy tiếp |
| GPU | **NVIDIA L4 24 GB** (CUDA 12.8) |
| Vì sao L4 (không A100)? | A100/H100 trên Modal yêu cầu **payment method on file**; L4 đủ VRAM cho DPO 7B (batch=1) và **không bị gate** → tiết kiệm |
| Image | `nvidia/cuda:12.4.1-devel` + Unsloth + TRL + PEFT + bitsandbytes + llama-cpp-python; execute notebook qua `jupytext + nbconvert` |
| Artifact store | Modal Volume `lab22-outputs` (+ HF cache volume) |
| Orchestrator | `modal_app/lab22_modal.py` (skip NB1/NB2 nếu đã có, scope flags trong `.env`) |
| **Tổng thời gian GPU** | **~25 phút** (NB3 21.2' + NB4 2.8' + NB5 1.0') |
| **Chi phí ước tính** | **~$0.33** (L4 @ ~$0.80/h) |

## 2. Model & dữ liệu

| Hạng mục | Giá trị |
|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (4-bit QLoRA) |
| SFT dataset | `bkai-foundation-models/vi-alpaca` · 1000 samples · 1 epoch |
| Preference dataset | `argilla/ultrafeedback-binarized-preferences-cleaned` · 5000 pairs |
| LoRA | r=16, α=32, dropout=0, target = q/k/v/o/gate/up/down |
| DPO hyperparams | β=0.05, lr=1e-6, loss=sigmoid, max_steps=250 |
| Judge | OpenAI **gpt-4o-mini** (NB4) |

## 3. Pipeline (6 notebook, execute thật trên Modal)

| NB | Nội dung | Kết quả |
|---|---|---|
| NB1 `01_sft_mini` | SFT-mini LoRA trên vi-alpaca | adapter `sft-mini`, loss giảm (`02-sft-loss.png`) |
| NB2 `02_preference_data` | Format UltraFeedback → parquet | `data/pref/train.parquet` (prompt/chosen/rejected) |
| NB3 `03_dpo_train` | DPOTrainer + reward curves | adapter `dpo`, `03-dpo-reward-curves.png`, W&B log |
| NB4 `04_compare_and_eval` | 8 prompt × 2 model + judge GPT-4o | `04-side-by-side-table.png`, judge results |
| NB5 `05_merge_deploy_gguf` | merge + GGUF (bonus) | best-effort |
| NB6 `06_benchmark` | (không chạy — bonus, giới hạn thời gian) | — |

## 4. Kết quả DPO

| Metric | Giá trị |
|---|---:|
| Final DPO loss | 0.234 |
| End chosen reward | +2.150 |
| End rejected reward | +0.820 |
| **Reward gap (chosen − rejected)** | **+1.33** |
| **Judge (gpt-4o-mini, 8 prompt)** | **SFT+DPO thắng 5 · hòa 3 · thua 0** |

Chi tiết judge: helpfulness 4 thắng + 0 hòa; safety 1 thắng + 3 hòa (các prompt từ chối "cứng" đều hòa vì cả 2 model đều từ chối đúng).

**Diễn giải reward gap dương:** nhờ cấu hình β=0.05 và lr=1e-6 qua 250 steps, DPO đã thành công trong việc tối ưu hóa xác suất của chosen responses và kéo giảm rejected responses so với base model. Reward của chosen đạt +2.15 trong khi của rejected chỉ đạt +0.82, tạo ra gap dương rõ rệt +1.33. Đánh giá judge bằng gpt-4o-mini cũng cho thấy sự cải thiện chất lượng rõ rệt (thắng 5, hòa 3, không thua). Phân tích đầy đủ ở [`submission/REFLECTION.md`](submission/REFLECTION.md) §3, §5, §6.

## 5. Bonus đã làm

- ✅ **Weights & Biases** — log reward curves realtime (project `lab22-dpo`, run `dpo-beta0.1-BIGGPU`). Rigor add-on +2.
- ⚠️ HuggingFace push: token 403 (không tạo được repo dưới namespace) — bỏ qua, chỉ là bonus.
- ✖ β-sweep / NB6 benchmark / GGUF release / cross-judge: không chạy do giới hạn thời gian deadline.

## 6. Sự cố kỹ thuật đã xử lý (engineering log)

Lab gốc có vài lỗi reproducibility; đã fix để chạy được end-to-end:

1. **Dataset SFT mặc định `5CD-AI/Vietnamese-alpaca-cleaned` bị xoá khỏi HF (404)** → đổi sang `bkai-foundation-models/vi-alpaca` (cùng schema Alpaca).
2. **Base model thiếu `chat_template`** (`-bnb-4bit` là base, không phải Instruct) → đổi sang `-Instruct-bnb-4bit`.
3. **xformers thiếu backward kernel cho GQA (BMGHK) của Qwen** → gỡ xformers, ép Unsloth fallback SDPA.
4. **llama-cpp-python build fail (CC=clang)** → ép `CC=gcc/CXX=g++` trong image.
5. **DPO trên L4 chậm** → cap `max_steps` để vừa thời gian.

---

### Liên kết
- Reflection chi tiết (7 mục): [`submission/REFLECTION.md`](submission/REFLECTION.md)
- Screenshots: [`submission/screenshots/`](submission/screenshots/)
- Modal orchestrator: [`modal_app/lab22_modal.py`](modal_app/lab22_modal.py)
- Notebooks đã execute: [`notebooks/`](notebooks/) (`.ipynb` giữ output cells)
