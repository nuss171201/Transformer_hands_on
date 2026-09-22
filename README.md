Built GPT2 from the ground up in PyTorch - 
    tokenizer, attention, full architecture, pretraining and two fine-tuning


things i implemented...

- BPE tokenizer, special tokens, sliding-window dataset with configurable stride
- token and positional embeddings
- self-attention from scratch -Q/K/V projections, scaled dot-product, causal masking, dropout
- multi-head attention with the batched weight split
- Layer norm and GELU, feed-forward block
- The full GPT model - embeddings, N transformer block, final norm output head 
- Text generation: greedy, temperature scaling, top-k sampling 
- Loading OpenAI's GPT2 weights - TensorFlow checkpoint parsing, fused QKV split
- Classification fine-tuning - spam classifier head, all but the last block frozen 
- Instruction fine-tuning - Alpaca prompts, custom collate with padding, target shift, '-100' masking
