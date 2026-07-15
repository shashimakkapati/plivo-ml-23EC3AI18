\#\# Experiment 6 – Increased Model Width

\*\*Hypothesis\*\*

The current best model (SwiGLU \+ RoPE) was using only \~1.31M of the allowed 2M parameters. Increasing model width while keeping the architecture unchanged should improve representational capacity and reduce validation bits-per-byte within the fixed 2000-step training budget.

\*\*Changes\*\*

\- Increased the embedding dimension (\`n\_embd\`) while keeping:  
  \- SwiGLU feed-forward network  
  \- Rotary Position Embeddings (RoPE)  
  \- Baseline optimizer  
  \- Baseline initialization  
  \- Byte-level tokenizer  
\- No changes were made to the training schedule or optimization algorithm.

\*\*Results\*\*

| Metric | Previous Best | Experiment |  
|--------|--------------:|-----------:|  
| Parameters | 1,314,880 | 1,724,080 |  
| Dev BPB | \*\*2.0925\*\* | \*\*2.0876\*\* |

\*\*Conclusion\*\*

Accepted. Increasing model capacity produced a measurable improvement in dev BPB while remaining comfortably under the 2M parameter limit. The gain was modest compared to the earlier SwiGLU and RoPE improvements, suggesting diminishing returns from simply scaling model size, but it established a new best-performing configuration.

\#\# Experiment 4 – SwiGLU Feed-Forward Network

\*\*Hypothesis\*\*

Replacing the standard GELU feed-forward network with a SwiGLU gating mechanism would improve the expressive power of the MLP while remaining within the parameter budget.

\*\*Changes\*\*

\- Replaced the two-layer GELU MLP with a SwiGLU block.  
\- Attention, optimizer, initialization and tokenizer remained unchanged.

\*\*Results\*\*

| Metric | Baseline | Experiment |  
|--------|---------:|-----------:|  
| Parameters | 1,339,840 | 1,335,360 |  
| Dev BPB | \*\*2.3718\*\* | \*\*2.3295\*\* |

\*\*Conclusion\*\*

Accepted. SwiGLU improved held-out BPB while slightly reducing the parameter count. This became the new best-performing architecture.

\#\# Experiment 5 – Rotary Position Embeddings (RoPE)

\*\*Hypothesis\*\*

Replacing learned positional embeddings with Rotary Position Embeddings would provide a stronger inductive bias for attention while reducing parameters.

\*\*Changes\*\*

\- Removed learned positional embeddings.  
\- Applied RoPE to query and key tensors.  
\- Kept the previously successful SwiGLU feed-forward network unchanged.

\*\*Results\*\*

| Metric | SwiGLU | SwiGLU \+ RoPE |  
|--------|--------:|--------------:|  
| Parameters | 1,335,360 | 1,314,880 |  
| Dev BPB | \*\*2.3295\*\* | \*\*2.0925\*\* |

\*\*Conclusion\*\*

Accepted. RoPE substantially improved held-out BPB while reducing parameter count. This became the best-performing model.

\#\# Experiment 3 – GPT-2 Style Initialization

\*\*Hypothesis\*\*

Replacing the baseline initialization with GPT-2 style initialization (smaller variance and residual projection scaling) would improve optimization stability and generalization.

\*\*Changes\*\*

\- Initialization std: 0.05 → 0.02  
\- Residual projection scaling: 0.02 / sqrt(2 × n\_layer)

\*\*Results\*\*

| Metric | Baseline | Experiment |  
|--------|---------:|-----------:|  
| Dev BPB | \*\*2.3718\*\* | \*\*2.5422\*\* |

\*\*Conclusion\*\*

Rejected. The GPT-2 initialization degraded performance under the strict 2000-step training budget. The larger baseline initialization appears to allow faster learning in this low-step regime.

\#\# Experiment 2 – Weight Tying

\*\*Hypothesis\*\*

Sharing the input embedding and output projection weights improves parameter efficiency and language modeling.

\*\*Changes\*\*

\- Enabled \`tie\_weights \= True\`.

\*\*Results\*\*

| Metric | Baseline | Experiment |  
|--------|---------:|-----------:|  
| Parameters | 1,339,840 | 1,298,880 |  
| Train loss | 1.7315 | 1.7651 |  
| Dev BPB | \*\*2.3718\*\* | \*\*2.4122\*\* |

\*\*Conclusion\*\*

Rejected. Weight tying reduced parameter count but slightly degraded held-out BPB under the fixed training budget. Reverted.

\#\# Experiment 1 – AdamW \+ Cosine Schedule

\*\*Hypothesis\*\*

Replacing Adam with AdamW and using linear warmup, cosine learning-rate decay, weight decay, and gradient clipping would improve optimization under the fixed 2000-step budget.

\*\*Changes\*\*

\- Adam → AdamW  
\- Weight decay \= 0.01  
\- Betas \= (0.9, 0.95)  
\- 5% linear warmup  
\- Cosine learning-rate decay  
\- Gradient clipping (max norm \= 1.0)

\*\*Results\*\*

| Metric | Baseline | Experiment |  
|--------|---------:|-----------:|  
| Train loss | 1.7315 | 1.9161 |  
| Dev BPB | \*\*2.3718\*\* | \*\*2.6270\*\* |

\*\*Conclusion\*\*

Rejected. Although the optimizer and schedule are commonly effective, they degraded held-out performance under the strict 2000-step budget. The baseline optimizer generalized better and these changes were reverted.  
