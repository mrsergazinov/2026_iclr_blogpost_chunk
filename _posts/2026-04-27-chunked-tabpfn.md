---
layout: distill
title: "ChunkTabPFN: Training-free Long Context"
description: Tabular foundation models such as TabPFN are limited in practice by the memory cost of attention, which grows quadratically with the number of samples and features. While efficient attention backends alleviate this in principle, CUDA grid limits and hardware compatibility gaps prevent their direct application at the scale of real-world tabular datasets. We introduce Chunked TabPFN, an exact tiling strategy that removes these implementation bottlenecks without retraining or approximation, extending TabPFN to 100K+ rows on a single GPU. On the long-context slice of TabArena, we find that — contrary to earlier reports — TabPFN's performance continues to improve with larger contexts, suggesting that the prior bottleneck was implementation-level memory, not model-level capacity.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous
authors:
  - name: Renat Sergazinov*
    affiliations:
      name: Department of Statistics, Texas A&M University
  - name: Shao-An Yin*
    affiliations:
      name: Department of Electrical and Computer Engineering, University of Minnesota
# *equal contribution note will appear in-body so we don't leak identity in metadata if anonymized version is needed

# must be the exact same name as your blogpost (no extension)
# and should exist at assets/bibliography/2026-04-27-chunked-tabpfn.bib
bibliography: 2026-04-27-chunked-tabpfn.bib

# Table of contents for the right-hand nav
toc:
  - name: 1. Introduction
  - name: 2. Methodology
  - name: 3. Experiments
  - name: 4. Conclusion

# Optional per-post styles (fine to delete if you don't need it)
_styles: >
  .inline-math {
    font-family: var(--sans-serif);
    font-weight: 500;
    background: rgba(0,0,0,0.03);
    padding: 0 .25em;
    border-radius: .25em;
  }
  .attn-figure-caption {
    font-size: 0.9rem;
    color: rgba(0,0,0,0.6);
    margin-top: 0.5rem;
    text-align: center;
  }

---

## 1. Introduction
<span id="sec:introduction"></span>

Large language models leverage **in-context learning (ICL)** by adapting their predictions at inference time based solely on provided examples, without requiring any gradient updates. Building on this idea, recent work on **tabular foundation models** — such as TabPFN, TabICL, Mitra, and Limix — extends the same paradigm to tabular data <d-cite key="hollmann2022tabpfn,hollmann2025accurate,qu2025tabicl,zhang2025mitra,zhang2025limix"></d-cite>. These models are trained once via stochastic gradient descent by optimizing the log-likelihood on synthetic tasks drawn from a prior, allowing them to approximate the posterior predictive distribution <d-math>p(y_{*} \mid x_*, D_{\text{train}})</d-math> in a single forward pass by supplying the training set as context <d-cite key="hollmann2022tabpfn,hollmann2025accurate"></d-cite>. This ICL-based approach is compelling because, unlike most deep tabular models — such as TabNet, FT-Transformer, NODE, and TabM, or retrieval-style models like TabR and ModernNCA — it requires no dataset-specific training or fine-tuning at deployment time <d-cite key="arik2021tabnet,gorishniy2021revisiting,popov2019neural,gorishniy2024tabm,gorishniy2023tabr,ye2024modern"></d-cite>.

ICL-based tabular models move closer to this ideal, but face a major practical limitation: **context length**. Transformer attention scales quadratically with sequence length, and current public TabPFN implementations are constrained to 10,000 samples<d-footnote>At the time of writing, the new TabPFN v2.5 model has just been released, which is supposed to have pushed the context limit further to 50,000.</d-footnote><d-cite key="hollmann2022tabpfn,hollmann2025accurate"></d-cite>. Many real-world tabular datasets, for example those collected in the TabRed becnhmark <d-cite key="rubachev2024tabred"></d-cite>, far exceed these limits.

To address this, researchers have experimented with **shrinking the context**, such as by clustering, partitioning, or retrieving only subsets of the data. Examples include random-forest partitioning <d-cite key="hollmann2025accurate"></d-cite>, the Mixture of In-Context Prompters (MICP) <d-cite key="xu2024mixture"></d-cite>, and KNN-style retrieval <d-cite key="thomas2024retrieval"></d-cite>. Others, like TuneTables <d-cite key="feuer2024tunetables"></d-cite>, compress the data into learned representations. While these methods can be effective, they come with two drawbacks:

- They often require **dataset-specific tuning** or even retraining, which contradicts the zero-shot, pure ICL philosophy.
- They do not use the **entire training set**, which is a core assumption of TabPFN’s Bayesian approximation. Replacing full data with summaries introduces conceptual inaccuracy.
- They introduce **additional hyperparameters and design choices** that may require careful tuning and can themselves become a source of variance in performance.

Hence, we ask the following question:

> Can we fit **all training examples** into the context (no pruning, no KNN) without learnable compression while staying within GPU memory?

In this work, we focus specifically on TabPFN, though we believe the conclusions extend to other ICL-based tabular models. We answer in the affirmative, though with important caveats. TabPFN's native implementation relies on FlashAttention <d-cite key="dao2022flashattention,dao2023flashattention,shah2024flashattention"></d-cite> and Memory-Efficient Attention <d-cite key="xFormers2022"></d-cite> — efficient attention mechanisms that reduce memory usage by fusing and tiling operations on the GPU rather than materializing the full attention matrix — but as we show, these backends fail in important cases.

- **TabPFN's attention structure is non-standard for these efficient attention backends**: Unlike language models, for which FlashAttention and Memory-Efficient Attention were originally designed, TabPFN applies attention both *between samples* and *between features*. This requires flattening leading dimensions into the batch dimension before each attention call, which can push the effective batch size well beyond 65,535 — the limit at which efficient CUDA kernels on Ampere-generation GPUs and below silently fail or crash with an `invalid configuration argument` error.
- **Efficient attention is often unavailable on consumer hardware**: Even setting aside the batch size issue, both backends are unsupported on older or consumer-grade GPUs — precisely the hardware most commonly used for tabular machine learning tasks — meaning that many practitioners cannot benefit from them without workarounds.

We **resolve both failure modes** with targeted changes to the attention computation. For efficient attention backends, we chunk inputs along the batch and head dimensions to stay within the 65,536 kernel limit. For older or consumer-grade GPUs where these backends are unavailable entirely, we implement a chunked forward pass in pure PyTorch using the incremental log-sum-exp trick to compute exact attention without materializing the full matrix or running out of memory. This patch yields results **identical to standard attention** (up to floating-point associativity), without any approximations, fine-tuning, or compression. 

Empirically, we test TabPFN's out-of-the-box scalability by evaluating it on the full **TabArena** benchmark <d-cite key="tabarena"></d-cite>, focusing specifically on datasets with **long contexts** (more than 10,000 rows). We report classification performance using the area under the curve (AUC) and regression performance using the root mean squared error (RMSE). Key findings include:

- **Accuracy improves with more data**: AUC and RMSE both continue to improve as dataset size grows, often up to 100,000 or more rows, suggesting that TabPFN benefits from larger training contexts rather than saturating early.
- **No degradation on smaller contexts**: On datasets with fewer than 10,000 rows, our chunked variant matches the performance of the original TabPFN, confirming that the extension introduces no hidden accuracy trade-offs at smaller scales.
- **Runtime remains practical**: Even on commodity GPUs, inference time stays within acceptable bounds, making large-context TabPFN viable outside of high-performance computing environments.

## 2. Methodology
<span id="sec:methodology"></span>

We use the following notation throughout this section. Let `B` denote the batch size (number of datasets), `L` the (padded) sample size, `F` the number of input features, `G` the number of feature groups, `D` the embedding dimension, and `H` the number of attention heads. Let `(X, y)` be the input to the TabPFN model with the feature tensor `X` having shape `[B, L, F]`. 

TabPFN groups and embeds the features, which yields a tensor of shape `[B, L, G, D]`.The labels `y` are similarly embedded and then concatenated with the features along the group dimension, producing an input of shape `[B, L, G + 1, D]`. Note that `y` and `X` have different logical lengths: `X` includes both train and test samples, while `y` is only provided for the training split. This is handled by padding the label embeddings for test samples with a dummy embedding. In the original TabPFN repository, a variable `single_eval_pos` in `transformer.py` holds the index at which train and test samples are split; readers inspecting the codebase will encounter this when tracing the train/test concatenation logic.

The core of TabPFN is the attention mechanism, implemented primarily in `layer.py`. TabPFN uses attention in two ways: **between samples** and **between features**. The between-sample attention has both self- and cross-attention components: self-attention among training samples and cross-attention from test samples to training samples. Following the TabPFN implementation, we assume attention layers expect input of shape `[B_eff, L, D]`, where `B_eff` is the effective batch size after flattening. In the code, the leading dimensions before `(L, D)` are collapsed via `_rearrange_inputs_to_flat_batch`. For between-feature attention, `B_eff = L * B`, whereas for between-sample attention `B_eff = (G + 1) * B`.

Efficient attention implementations in PyTorch (such as the fused CUDA kernels backing `torch.nn.functional.scaled_dot_product_attention`) tile work across `B_eff` and `H`. On NVIDIA GPUs of Ampere architecture and below, this limits the product `B_eff * H` to at most `65535` CUDA blocks; at `65536` the kernel fails with `CUDA error: invalid configuration argument`. This is a known issue, reported independently on the [PyTorch](https://github.com/pytorch/pytorch/issues/133976), [xFormers](https://github.com/facebookresearch/xformers/issues/845), and [FlashAttention](https://github.com/Dao-AILab/flash-attention/issues/1786) repositories. The PyTorch issue includes a minimal reproducer where `65535` succeeds but `65536` fails. In TabPFN, large `L` or `G` can easily push `B_eff` past this limit.

A simple practical fix is to loop over the flattened batch dimension in chunks, so that each call to `scaled_dot_product_attention` stays within the kernel’s limits. This keeps the rest of the model unchanged while avoiding the `invalid configuration` errors at large `L` or `G`. Conceptually, this is can be done via the following patch to the attention computation.

```python
outputs = []

for q_chunk, k_chunk, v_chunk in zip(
    torch.split(q, batch_size, dim=0),
    torch.split(k_b, batch_size, dim=0),
    torch.split(v_b, batch_size, dim=0),
):
    # (B_chunk, Lq, H, D) -> (B_chunk, H, Lq, D)
    Q = q_chunk.permute(0, 2, 1, 3).contiguous()
    K = k_chunk.permute(0, 2, 1, 3).contiguous()
    V = v_chunk.permute(0, 2, 1, 3).contiguous()

    out = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=dropout_p if dropout_p is not None else 0.0,
        is_causal=False,
        scale=softmax_scale,
    )  # (B_chunk, H, Lq, D)

    # -> (B_chunk, Lq, H, D)
    outputs.append(out.permute(0, 2, 1, 3).contiguous())

attention_head_outputs = torch.cat(outputs, dim=0)
```

A  different issue is **hardware support** for efficient attention kernels. PyTorch's `scaled_dot_product_attention` can dispatch to several backends on CUDA: FlashAttention, memory-efficient attention, or a plain math fallback. The availability of specialized kernels varies across GPU generations. For older or unsupported devices, we provide a pure-PyTorch reimplementation of memory-efficient tiling based on [this repository](https://github.com/lucidrains/memory-efficient-attention-pytorch/tree/main). The key idea is to tile both queries and keys/values so that only a small `l x r` block of the score matrix is materialized at a time, while maintaining numerically stable running statistics via a log-sum-exp merge. We provide a brief sketch below.

```python
def chunked_attention(q, k, v, q_chunk, kv_chunk, scale):
    """
    q: (..., Lq, D)
    k: (..., Lk, D)
    v: (..., Lk, Dv)
    q_chunk: size of query tiles (l)
    kv_chunk: size of key/value tiles (r)
    """
    Lq, Lk = q.size(-2), k.size(-2)
    outputs = []

    for qs in range(0, Lq, q_chunk):
        qe = min(qs + q_chunk, Lq)
        q_tile = q[..., qs:qe, :]                            # (..., l, D)

        # running stats per query row
        mu = q_tile.new_full(q_tile.shape[:-1], -float("inf"))  # (..., l)
        s  = torch.zeros_like(mu)                               # (..., l)
        a  = torch.zeros(*mu.shape, v.size(-1),
                         device=q.device, dtype=q.dtype)        # (..., l, Dv)

        for ks in range(0, Lk, kv_chunk):
            ke = min(ks + kv_chunk, Lk)
            k_tile = k[..., ks:ke, :]                           # (..., r, D)
            v_tile = v[..., ks:ke, :]                           # (..., r, Dv)

            logits = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale
            local_max = logits.max(dim=-1).values               # (..., l)
            new_mu = torch.maximum(mu, local_max)

            # rescale old aggregates
            alpha = torch.exp(mu - new_mu)
            s *= alpha
            a *= alpha[..., None]

            # accumulate current tile
            exp_logits = torch.exp(logits - new_mu[..., None])
            s += exp_logits.sum(dim=-1)                         # sum_k e^{z_k}
            a += torch.matmul(exp_logits, v_tile)               # sum_k e^{z_k} v_k

            mu = new_mu

        outputs.append(a / s[..., None])                        # softmax = a / s

    return torch.cat(outputs, dim=-2)                           # (..., Lq, Dv)
```

In this implementation, queries are tiled into chunks of size `l` (controlled by `q_chunk`) instead of processing all `Lq` at once. Keys and values are streamed in chunks of size `r` (controlled by `kv_chunk`), so only an `l x r` block of logits is materialized at a time. Per-row running statistics `(mu, s, a)` are maintained via a numerically stable log-sum-exp merge, ensuring the final output matches full attention exactly — as if the entire `Lq x Lk` score matrix had been formed in one pass.

Having described both chunking strategies, we now address their **interference with the rest of the TabPFN architecture**. Both the chunking over `B_eff` and the memory-efficient tiling operate strictly within the attention computation. In PyTorch, each layer module — including LayerNorm, residual connections, and any positional encodings — is encapsulated: its forward pass depends only on its input tensor, not on how upstream activations were produced. The **chunked attention yields the same output tensor as the unchunked variant, up to floating-point associativity**, so all downstream modules behave identically. It is worth emphasizing that the original TabPFN already relied on hardware-level tiling via efficient attention backends (e.g., FlashAttention). Our contribution adds two layers on top: (1) chunking along `B_eff` for efficient kernels to respect CUDA grid limits, and (2) a pure-PyTorch reimplementation of memory-efficient tiling for devices that lack fused kernel support. Neither modification alters the mathematical operation performed; both are purely computational changes. We verify empirical equivalence on all dataset sizes where the baseline TabPFN is feasible (Section 3). We acknowledge that correctness at longer contexts is necessarily extrapolated from these results and from the mathematical invariance of the tiling scheme.

While the above establishes correctness, the chunking approach does have **practical performance implications**. Chunking converts some parallel execution into sequential processing, introducing a trade-off between memory scalability and inference latency. For large datasets — where either `L` or `G` pushes `B_eff` well beyond the kernel limit — wall-clock time grows approximately linearly with the number of chunks. This can make single-GPU inference slow when predicting many test points simultaneously. However, because chunks are independent, the workload is straightforwardly distributable across multiple GPUs, offering a path to recovering parallelism without sacrificing the memory benefits. We analyze this trade-off empirically in Section Y, reporting both peak memory and wall-clock time as a function of dataset size.

## 3. Experiments

<span id="sec:experiments"></span>

We evaluate the TabPFN v2 model with chunking enabled on **TabArena** <d-cite key="tabarena"></d-cite>, which includes 51 tabular datasets spanning classification and regression tasks. We report scaling statistics for memory and runtime in Figure 1 and the overall performance on TabArena in Figure 2. Note that in the original and subsequent reports of TabPFN, LIMIX, and TabICL on TabArena, the authors have typically imputed values that exceeded the context length for their respective methods. This might have created a distorted view of model capabilities. In Figure 2, we use only directly measured (non-imputed) results.

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/tabarena_long_results.png" class="img-fluid" %}

<div class="attn-figure-caption"> Figure 1. Scaling TabPFN v2 to long contexts. Chunked TabPFN matches baseline accuracy where both fit, and extends inference to 100K+ examples. </div> 

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/elo_vs_baselines.png" class="img-fluid" %}

<div class="attn-figure-caption">
Figure 2. Elo and normalized score across TabArena. Striped bars denote prior TabPFN results where out-of-memory failures were imputed using Random Forest fallbacks. Solid bars show our chunked TabPFN, which runs all datasets directly without fallbacks.
</div>

Separately, we evaluate how context length affects TabPFN v2's performance on the 15 "long-context" datasets from TabArena. For each dataset, we randomly subsample the training set to progressively larger sizes (3,000 → 5,000 → 10,000 → 20,000 → 50,000 → 100,000) while keeping the test set fixed, and compare baseline TabPFN v2 against our Chunked TabPFN. We report performance, memory, and runtime in Figure 3.

* Chunked TabPFN maintains *exact equivalence* to baseline TabPFN while extending feasible context length by roughly 10×.
* Empirical scaling shows either plateau or monotonic improvement—never catastrophic degradation.
* Memory and runtime growth are linear in chunk size, enabling inference on 100 K+ examples with a single GPU.

These findings reinforce that **TabPFN's in-context generalization truly extends beyond its training limit**, and that the primary bottleneck was *implementation-level memory*, not *model-level capacity*.

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/tabarena_long_results_per_dataset.png" class="img-fluid" %}

<div class="attn-figure-caption">
Figure 4. Scaling curves for the long-context datasets. Each plot shows RMSE or AUC (left axis), wall-clock inference time in seconds, and peak GPU memory in MB. Chunked TabPFN tracks baseline accuracy exactly up to 10K examples and continues scaling to 100K mostly without degradation.
</div>

## 4. Conclusion

<span id="sec:conclusion"></span>

We presented **Chunked TabPFN**, an exact tiling strategy that enables TabPFN to process *long-context* tabular datasets (100K+ rows) without retraining, fine-tuning, or any pre-processing such as clustering or compression.

The chunked attention computation is mathematically identical to the original transformer attention — only the evaluation order changes. Predictions match baseline TabPFN within floating-point tolerance for all short-context cases. Because peak GPU memory scales linearly with tile size rather than quadratically with context length, the practical 10K-sample ceiling is removed, allowing inference on 100K+ rows using 24–32 GB GPUs.

Crucially, Chunked TabPFN retains the spirit of in-context learning: no dataset-specific training, no hyperparameter search, no adaptation steps. Despite this simplicity, it matches or surpasses tuned deep tabular models on the long-context slice of TabArena. Moreover, many datasets continue to improve with larger contexts, suggesting that the PFN prior generalizes beyond its nominal pre-training length.