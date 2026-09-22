# Build a Large Language Model (From Scratch)

My implementation while working through Sebastian Raschka's
[*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch)
(Manning, 2024). Every component written by hand in PyTorch — no Hugging Face `transformers`.

## Contents

| Notebook | Topic |
|---|---|
| `01_working_with_text_data` | BPE tokenization, sliding-window dataset, token and positional embeddings |
| `02_coding_attention_mechanisms` | Self-attention, causal masking, multi-head attention |
| `03_implementing_a_gpt_model` | Layer norm, GELU, feed-forward, transformer blocks, full GPT |
| `04_pretraining` | Training loop, cross-entropy loss, loading OpenAI's GPT-2 weights |
| `05_fine_tuning_for_classification` | Spam classifier on the SMS Spam Collection |
| `06_fine_tuning_to_follow_instructions` | Instruction fine-tuning, Alpaca prompt format, LLM-as-judge evaluation |

`src/GPT_from_scratch.py` holds the consolidated model and training code the notebooks import.

## Results — instruction fine-tuning

Fine-tuned GPT-2 medium (355M) on 935 instruction–response pairs for 2 epochs.

Evaluated by scoring all 110 held-out test responses with Llama 3 8B running
locally via Ollama: **average 56.2 / 100**.

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

Pretrained GPT-2 weights download automatically on first run (~1.4 GB for the
medium model). Trained checkpoints are gitignored — rerun the notebooks to
regenerate them.

## Note: MPS memory blowup on Apple Silicon

Training on MPS with per-batch dynamic padding caused PyTorch's caching
allocator to hold a separate cached block for every distinct sequence length.
Since MPS allocations are wired and unified memory is shared with the CPU, the
footprint grew until the 24 GB was exhausted and the machine started swapping —
GPU utilization collapsed while the CPU sat 85% idle.

Fixing the pad length at 128 so every batch has identical shape let the
allocator reuse one block. Training went from ~58 minutes to ~14.

## Attribution

Implementations follow the book closely. `src/gpt_download.py` is Raschka's
GPT-2 downloader, used unmodified. `data/the-verdict.txt` is Edith Wharton's
"The Verdict" (1908), public domain.

Original repository: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0).
