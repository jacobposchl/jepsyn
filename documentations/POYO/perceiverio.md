# POYO: Multi-Session Neural Data Handling — Implementation Synthesis

## The Core Problem

Each neural recording session has a unique, non-overlapping set of neurons. You cannot align neuron #47 from Monkey C with neuron #47 from Monkey M — they are biologically distinct entities with no known correspondence. Traditional approaches that treat neurons as fixed input channels (like a fixed-size vector of firing rates) completely break down across sessions.

POYO's solution is to never treat neurons as fixed channels. Instead, it treats **individual spikes as tokens**, making the neuron identity a learned property rather than a fixed structural assumption.

---

## 1. The Tokenization Scheme

This is the foundational design decision everything else builds on.

Each spike becomes a token defined by two things: **what unit fired** and **when it fired**.

```
token_i = (UnitEmbed(unit_id), timestamp_i)
```

`UnitEmbed` is a simple learned lookup table — essentially an embedding matrix where each row corresponds to one known unit. For a session with 150 units, you have 150 rows. For a session with 400 units, 400 rows. There is **no shared structure assumed between the rows of different sessions**.

Key implementation details:

- All spikes from the same unit share the **same embedding vector** — they differ only in their timestamp
- The context window is fixed in **time** (1 second in the paper), not in token count — so the number of tokens M varies depending on how many neurons are present and how fast they fire
- **Delimiter tokens** are added per unit per window: a `[START]` and `[END]` token at timestamps `t` and `t+T`, computed as `delimiter_embedding + unit_embedding`. This is critical because without it, a silent unit (one that doesn't fire in the window) would be completely invisible to the model
- During batching, sequences are padded to the longest in the batch, with an attention mask applied in the first cross-attention layer to ignore padding

---

## 2. The Architecture

The architecture is a **PerceiverIO** variant with three stages: compress, process, query.

### Stage 1 — Cross-attention compression (spikes → latents)

Rather than running self-attention over all M spike tokens (which is O(M²) and M can reach 20k), a set of **N learned latent tokens** Z₀ attend to the spike tokens via cross-attention:

```
Q = W_q · Z₀          # queries from latent tokens
K = W_k · X           # keys from spike tokens  
V = W_v · X           # values from spike tokens

Z₁ = softmax(QKᵀ / √d_k) · V
```

N is fixed at 512 (or 256 for single-session models). This compresses an arbitrarily long spike sequence into a fixed-size latent representation.

The latent tokens are **grouped and spread evenly across the time window**. If you have 256 latents divided into groups of 8, you get 32 groups with timestamps `t₀, ..., t₃₁` distributed uniformly over [0, T]. Tokens within a group share the same learned initial embedding (weight sharing as an inductive bias), but get different timestamps and evolve independently after the first cross-attention.

### Stage 2 — Self-attention in latent space

L layers of standard transformer self-attention operate on the N latent tokens:

```
Z_{l+1} = Self-Attn(Z_l, Z_l, Z_l)
```

Cost is O(N²) instead of O(M²). In the paper, N=512, M can be up to 20k, so this is a ~1600x reduction in attention cost.

### Stage 3 — Cross-attention readout (latents → outputs)

Output query tokens Y₀ attend to the final latent sequence Z_L:

```
Y = softmax(Q_Y · Z_Lᵀ / √d_k) · Z_L
```

The number of output tokens P equals the number of behavioral timepoints you want to predict, which can differ across datasets (due to variable sampling rates). The paper handles this by randomly sampling 100 timepoints per window during training to avoid over-representation of high-rate datasets.

Each output token is initialized to the **same learned embedding** plus a **session embedding** (described below), and is assigned a timestamp corresponding to the behavioral timepoint it's predicting. The output then passes through a linear/MLP head to produce the behavioral variable (e.g., 2D hand velocity).

---

## 3. Handling Variable Sessions: The Session Embedding

To account for hidden experimental variables that differ between sessions (recording equipment, task version, preprocessing pipeline, etc.), each session gets its own **learned session embedding** injected into the output query:

```
y₀ᵢ = y₀_base + SessEmbed(session_id)
```

This is another lookup table. When you encounter a new session, you add a new row. The rest of the model is unchanged. Think of the query as asking: *"Predict [BEHAVIOR] at [TIME] under the experimental conditions of [SESSION_ID]."*

The paper shows that these session embeddings cluster meaningfully in PCA — sessions from the same lab group together without explicitly being told lab identity — which validates that they're capturing real experimental structure.

---

## 4. Rotary Position Encoding (RoPE) for Timing

Every token (spike, latent, output) has an associated timestamp. Timing is encoded via **RoPE** applied in all attention layers, making attention scores depend on **relative time differences** between tokens rather than absolute positions.

The rotation matrix for a timestamp t is:

```
R(t) = block_diag([R₂ₓ₂(t, T₀), R₂ₓ₂(t, T₁), ..., R₂ₓ₂(t, T_{D/2-1})])
```

where T_i are log-spaced periods between T_min=1ms and T_max=4s, covering the full range of relevant neural timescales.

The attention score becomes:
```
a_ij = softmax((R(t_i)·q_i)ᵀ · (R(t_j)·k_j))
```

Since `R(tᵢ)ᵀ · R(tⱼ) = R(tⱼ - tᵢ)`, the score depends only on relative timing.

The paper also applies RoPE to **values**, not just queries/keys — an extension that lets the output of each attention block carry explicit timing information into the feedforward layers:

```
z_j ← z_j + R(-t_j) · Σᵢ [a_ij · R(t_i) · v_i]
```

Implemented efficiently by pre-rotating all values before attention, then post-rotating outputs. They also found that rotating only **half** the head dimensions (replacing the other half with identity) improved performance.

---

## 5. Adapting to New Sessions: Unit Identification

This is the key transfer mechanism. When you encounter a new recording with neurons the model has never seen:

1. **Freeze all existing model weights**
2. Add new rows to `UnitEmbed` for each new unit (randomly initialized)
3. Add a new row to `SessEmbed` for the new session
4. Run gradient descent on **only these new embeddings**, using whatever labeled data you have

This converges very quickly — under a minute on a single GPU, or a few minutes on CPU. The rest of the model (all the transformer weights) is completely untouched.

The paper validates this by taking units that *were* seen during pretraining, wiping their embeddings, and re-running unit identification. After 400 epochs, the recovered embeddings have cosine similarity ~0.8–0.85 with the original pretrained embeddings, and nearest-neighbor accuracy of ~97–100%. This confirms the latent space is genuinely meaningful and recoverable.

If you have more data and want better performance, you can follow unit identification with **full finetuning** (gradual unfreezing: first train only unit/session embeddings, then unfreeze all weights). The paper uses LAMB optimizer, lr=1e-4, weight decay=1e-4, batch size=128 for finetuning.

---

## 6. Training Across Sessions

**Loss:** MSE over hand velocity (z-scored per lab). During center-out tasks, reaching segments are upweighted by 5x to focus on the most informative neural activity.

**Data augmentation:** Unit dropout — randomly subsample a subset of units from the population during training (minimum 30 units). This forces the model to be robust to missing neurons and is critical for generalization.

**Load balancing across GPUs:** Since M (spike token count) varies from ~1k to ~20k, sequences are bucketed by length and distributed so that each GPU gets sequences of similar length, with local batch size adjusted accordingly. This minimizes wasted padding compute.

**No data standardization across labs:** They deliberately do not re-sort spikes, resample behavior, or homogenize preprocessing across datasets. The model must handle raw diversity. The only processing applied is behavioral outlier clipping based on acceleration.

---

## 7. Implementation Summary Checklist

| Component | What to implement |
|---|---|
| Unit embeddings | Lookup table; new rows per session/recording |
| Session embeddings | Lookup table; one entry per session |
| Spike tokenizer | (UnitEmbed + timestamp) per spike; add [START]/[END] delimiter tokens per unit |
| Perceiver encoder | Cross-attention: learned latents attend to spike tokens |
| Latent processor | L layers of self-attention on N fixed latent tokens |
| Output decoder | Cross-attention: output query tokens attend to latents |
| Position encoding | RoPE on all attention layers (queries, keys, AND values); half-dimension rotation |
| Transfer to new session | Freeze model; gradient descent on new unit + session embeddings only |
| Finetuning | Gradually unfreeze: unit/session embeds first, then full model |
| Training augmentation | Unit dropout (keep min 30 units) |
| Batching | Time-based windows (1s); variable M; pad to longest in batch with attention mask |

The key architectural insight to internalize: the model **never sees a fixed-size neural vector**. It always operates on an event sequence. Session-specific information lives entirely in the embedding lookup tables, which are cheap to extend, freeze, or optimize independently from the rest of the model.