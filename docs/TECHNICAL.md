# Technical reference

[Back to the quick-start README](../README.md)

This document covers checkpoint compatibility, execution paths, setup details,
and diagnostics for Prompt Refiner. Start with the README for installation and
normal use. Support for a storage layout does not guarantee that every export,
model variant, or GPU backend has been tested end to end.

- [Model compatibility](#supported-models)
- [Hardware and quantization choices](#hardware-and-quantization-choices)
- [Dependencies](#dependencies)
- [Controls, settings, and presets](#controls-settings-and-presets)
- [Sampling and presence penalty](#sampling-and-presence-penalty)
- [Qwen3.5 / MOSS-Image](#qwen35-9b--moss-image)
- [Architecture configuration and tokenizers](#architecture-configuration-and-tokenizers)
- [Safetensors formats](#safetensors-formats-and-memory)
- [GGUF](#gguf)
- [Validation coverage and limitations](#validation-coverage-and-limitations)
- [Memory reporting](#memory-reporting)
- [Missing LM heads](#missing-lm-heads)
- [Uninstall and stored files](#uninstall-and-stored-files)

## Supported models

| Language model | Typical Neo image model | Notes |
| --- | --- | --- |
| Qwen3 | Z-Image, Flux.2 Klein, Anima | Known dense 0.6B / 1.7B / 4B / 8B / 14B / 32B profiles; base models may follow instructions poorly. |
| Qwen3-VL | Krea2, Qwen-Image 2.1 | Known dense 2B / 4B / 8B / 32B profiles; only language weights are loaded. |
| Qwen2.5-VL 7B | Qwen-Image / Qwen-Image-Edit | Preserves biased Q/K/V projections; requires its original output head if omitted. |
| Qwen3.5 9B (dense) | MOSS-Image | Text-only hybrid Gated DeltaNet/full-attention decoder; safetensors/GGUF; requires its original output head if omitted. |

Inputs are text only. Vision weights and `mmproj` are not needed for refinement.
MoE (including Qwen3.5 MoE), Gemma, Ministral, CLIP, and T5/UMT5 are not supported
by this version. Selecting a file does not make an unsupported quantization work:
unknown layouts fail with an error before generation.

Prefer an instruction-tuned model for prompt rewriting. The extension cannot turn
base-model weights into an instruction-tuned model. Base checkpoints can work, but instruction following depends on the checkpoint
and chat template. Anima-identified Qwen3 files prefer Anima's own chat template
among Forge's bundled tokenizers. Base filenames produce an advisory status warning. **Disable thinking via chat
template** applies to switchable Qwen3 and Qwen3.5; when needed, it adds the standard empty thinking
block. Completed and unfinished thinking blocks are hidden from displayed output. Thinking
models can require a larger output token budget. Dedicated Thinking-2507 models
are not switchable thinking/non-thinking models: the checkbox cannot reliably
disable their reasoning. For filenames identifying Thinking-2507, the extension
ignores that checkbox and reports the limitation in the status and console.
Prefer a standard or Instruct checkpoint for this task.

## Hardware and quantization choices

The GPU path targets NVIDIA CUDA through Forge Neo's installed PyTorch and
quantization backends. No GPU model name or fixed VRAM size is required. Automatic
device selection uses an available GPU; **Compute dtype: auto** selects BF16 when
supported, otherwise FP16 on the GPU, and BF16 on the CPU. Kernel availability
also depends on the installed packages, tensor layout, and GPU capabilities.

Choose an encoder for the memory available during refinement, leaving room for
dense embeddings, the output head, caches, and temporary decoded layers. The
checkpoint's disk size alone is not a runtime memory estimate. A card with more
VRAM allows more headroom or larger encoders; it does not guarantee a particular
speed or that every 9B/32B export will fit. **Unload Forge's image models from VRAM**
can free memory before loading the encoder. Keeping the refiner loaded occupies
memory between runs.

- FP8 and INT8/ConvRot are useful starting points when the encoder fits comfortably.
  Supported FP8 storage does not require native FP8 matrix multiplication: the
  extension's fused single-token path reads compressed values directly, and other
  operations can use decoding fallbacks.
- W4A8 reduces packed weight storage for larger encoders. W6A8 uses about 1.5 times
  the packed code bytes for greater weight precision; scales and dense tensors add
  overhead to both. Their execution depends on the installed Comfy-Kitchen backend.
- INT4 ConvRot further reduces resident weight storage. Its native W4A4 path also
  quantizes activations, so check the generated text for quality.
- NVFP4 uses native FP4 operations only on eligible Blackwell GPUs with a working
  backend. Earlier CUDA GPUs use per-layer decoding: this can save resident memory
  but requires a dense temporary and does not provide native FP4 execution.
- GGUF Q4_K_M is a practical compatibility option; Q5/Q6 trade more storage for
  weight precision. Neo decodes the active layer, so GGUF speed can differ greatly
  from a dedicated LLM inference engine.

For example, RTX 3060 12 GB, RTX 4060 8 GB, and RTX 5070 Ti 16 GB should be evaluated
by available memory and supported execution paths, rather than treated as separate
extension profiles. These examples are not a claim that all three have been tested.
The local GPU checks described below were run on RTX 4060; they establish that
configuration's coverage, not a requirement or benchmark for other cards.

Automatic CPU/GPU layer offloading is not implemented. Windows shared-memory
paging may allow a model to load beyond physical VRAM, but generation can become
extremely slow; other systems may instead report an out-of-memory error. Choose a
smaller encoder or stronger compression, or select **Device: cpu** if necessary.
CPU inference is available but can be much slower, particularly for large models.
AMD/ROCm is untested and is outside the currently validated GPU support scope.

## Dependencies

Use an up-to-date Forge Neo installation with Qwen3 support in Transformers.
The extension reuses Forge's packages and does not install or upgrade them.
GGUF needs Neo's existing `gguf` package and tensor operations. W4A8/W6A8,
INT4 ConvRot, and NVFP4 require the corresponding Comfy-Kitchen layouts;
W6A8 specifically needs the 6-bit-capable `AsymW4A8Int8Layout` implementation.
Qwen3.5 uses the extension's local decoder with Forge's Transformers 4.x packages.

## Controls, settings, and presets

The refiner is an accordion in txt2img and img2img. The idea and result are side
by side, with refinement controls below the idea and **Apply to prompt** below the
result. An empty idea uses the tab's current prompt with extra-network tags
(`<...>`) removed; applying the result keeps the current prompt's tags that the
result does not already contain. Preset selection, **Max new tokens**, and
**Disable thinking** are below them; disabling thinking is checked by default for
switchable Qwen3 models. Both tabs share one loaded model and one `settings.json`;
a change saved in one tab is shown in the other after Forge restarts. Expand **System prompt
& presets** to edit instructions or manage presets. **Sampling options** and
**Advanced model settings** are collapsed by default. The short status appears
below the result; expand **Model details** for tokenizer, quantization, device,
and memory diagnostics or to unload the model manually.

Selections and options are saved automatically on change to `settings.json` in
the extension folder and restored after a page reload or Forge restart. These
controls opt out of Forge's separate `ui-config.json` defaults so they cannot
overwrite the extension's saved choices or restore excluded text fields. This
includes the encoder, selected system prompt preset, checkboxes, sampling/model
options, tokenizer directory, and LM head choices. A missing model, preset, or
head falls back to the available default. Invalid saved values use defaults;
settings read/write failures produce a nonfatal message in **Settings status**.
Your prompt/idea, generated result, unsaved system prompt edits, preset-name draft,
and Hugging Face token are never saved in this file. System instructions are
saved only through **Save preset**. Delete `settings.json` while Forge is stopped
to reset preferences.

The model is unloaded after each run by default. **Keep model loaded** avoids
reloading at the cost of occupied memory. **Unload model** releases it manually.
After a loading or generation error, the refiner always unloads, including when
**Keep model loaded** is enabled. It stops and joins any active generation worker,
releases input/model references, and clears failed traceback frames before garbage
collection and returning unused CUDA cache. Other Forge models and GPU programs
can still occupy VRAM; this releases the failed refiner run's allocations.
Loading from disk is reported whenever needed; retaining a model only avoids
reloading when the selected encoder and model settings still match.
Presets are editable in the UI and saved in `presets.json`. Selecting a preset
fills both its instructions and name. **Save preset** updates that name; change
the name to save a copy. **Delete selected preset** removes the selected entry,
including the last entry if desired. Validation and file errors appear in
**Preset status** without a console traceback, and failed saves preserve the editor.
Before each save or deletion, an existing `presets.json` is copied to
`presets.json.bak`, replacing the previous backup. To undo the last saved change,
stop Forge, copy `presets.json.bak` over `presets.json`, and restart Forge. This is
one previous version, not a backup history. Malformed files are never silently
replaced with defaults. Both files are inside the extension folder.
An empty prompt or missing encoder produces a status message without loading a
model or raising an exception. The console logs skipped requests, the start of
refinement, completion/stopping, and failures; prompt text is not logged.

### Sampling and presence penalty

See the [README sampling table](../README.md#sampling-recommendations) for profiles.
New installations use the Qwen3-VL 4B Instruct VL model-card values, including
presence penalty 1.5, with min-p disabled. Existing settings are preserved; the reset
button changes only the six sampling controls, whose change handlers persist them.
Model selection does not import a generation config or apply another profile.

The presence control is implemented as an additive Transformers logits processor
because Forge's Transformers 4.x runtime does not provide that generation option.
For each sequence, it subtracts the selected value from each token ID appearing
anywhere after the complete encoded prompt prefix, once regardless of frequency.
System instructions, the user's idea, and the chat-template prefix are excluded.
It works with both cached and uncached generation and with greedy or sampled
selection; it preserves the score dtype and does not modify scores in place.
This is a token-level penalty, not a detector for repeated phrases or concepts.

The generation request explicitly overrides temperature, top-p, top-k, min-p,
and repetition penalty. Disabled filters cannot be restored from model defaults.
At temperature zero, top-p/top-k/min-p are ignored; repetition and presence
penalties remain active. Presence penalty is saved in `settings.json` like the
other sampling controls and defaults to 1.5 when older settings omit it. Explicitly
saved values, including zero, are preserved.

## Qwen3.5 9B / MOSS-Image

Select a dense Qwen3.5 9B `.safetensors` or standard `qwen35` `.gguf` encoder in the usual encoder directory.
**Model family: Auto** recognizes its hybrid tensor layout; an explicit **Qwen3.5**
selection is also available. The published 9B dimensions are inferred without an
extra config. Original multimodal HF configs with `text_config` are accepted.
Other Qwen3.5 sizes and MoE checkpoints are outside this version's advertised scope.

A local text-only decoder runs with Forge's existing Transformers 4.x packages;
no upgrade, additional language model download, vision tower, or external server
is required. It preserves the 24 Gated DeltaNet layers, eight gated full-attention
layers, zero-centered RMSNorm, partial rotary positions (10M theta), and separate
output head. Generation caches convolution/recurrent states and attention KV per
request, then discards them; keeping model weights loaded never shares conversation
or recurrent state between prompts. Prompt processing uses a chunked delta-rule
solver, while generated tokens use recurrent updates with FP32 state.

Use Neo's bundled `backend/huggingface/OpenMOSS-Team/MOSS-Image-SFT/tokenizer`,
or put the matching local tokenizer in this extension's `tokenizer/qwen3_5/`.
Qwen3/VL tokenizer files are incompatible: Qwen3.5 uses a different vocabulary
and special-token IDs. Encoding-only tokenizer configs use a text ChatML fallback.
The thinking checkbox also applies to Qwen3.5 and defaults to disabled thinking.
It requests an empty thinking block; fine-tunes can still differ in instruction following.

BF16/FP16/FP32, scaled INT8/ConvRot, row/tensor-scaled FP8, W4A8/W6A8,
INT4 ConvRot, and NVFP4 linear weights use the validated storage adapters. The small convolution, decay, and norm
parameters must use a supported dense layout. Vision and MTP (multi-token prediction)
weights, if present, are skipped. Standard llama.cpp Qwen3.5 GGUF exports use
the hybrid mapping described below.
Embeddings and convolution weights use the selected compute dtype, including when
GGUF stores them as FP32. FP32 normalization weights are retained where present,
and recurrent updates still use FP32 with BF16/FP16 activations.

The official `OpenMOSS-Team/MOSS-Image-SFT/text_encoder` index contains four
safetensors shards with the `language_model.` prefix and no `lm_head` tensor.
Export/merge those shards into a single supported safetensors encoder for this
extension; it does not discover HF index files or combine shards automatically.
The official prefix and Comfy-style `model.language_model.` prefix are both
recognized. Community exports may include an output head, but this is not assumed.

If `lm_head.weight` is present, it is used directly. If it is absent, obtain the
head from the exact original checkpoint (for standard weights, `Qwen/Qwen3.5-9B`)
using the existing **LM head** controls. A head from another fine-tune is not
interchangeable merely because it has the same dimensions.

A 9B INT8 encoder also needs memory for dense embeddings, its output head, and
recurrent/KV state. Compression does not guarantee that it fits in available VRAM. Use **Device: cpu** or a GPU
with sufficient memory; automatic CPU/GPU layer offloading is not implemented.
The recurrence uses PyTorch, so speed differs from a specialized LLM inference engine.
Validation includes full CPU generation with a community 9B INT8 ConvRot export,
small cached/quantized models on CPU and GPU, numerical comparisons with Neo's
attention implementations, and a headless official-style prefix with a sidecar head.
The full official four-shard checkpoint has not been downloaded or tested end to end.

## Architecture configuration and tokenizers

Architecture dimensions are checked against the tensors. The model family comes
from an explicit selection, an adjacent Hugging Face config, GGUF architecture
metadata, or recognizable checkpoint signatures/names. Ambiguous names produce
an actionable error. Automatic RoPE uses the supplied config/GGUF metadata first,
then the family default (10M for Qwen3.5, 5M for Qwen3-VL and named Qwen3 2507 refreshes, 1M for
older Qwen3 and Qwen2.5-VL). The RoPE control remains available as an override.

For a renamed checkpoint or a configuration outside the known profiles, place its
original HF config beside it as `<encoder-stem>.config.json`; a neighboring
`config.json` is also accepted. Complete VL configs containing `text_config` work.
The config must match the tensors and specify `tie_word_embeddings` for unknown
profiles. An unknown signature is treated as a setup issue: **Status** and a short
console line show the required config, without a traceback. The existing result
is preserved, and model resources are released even with **Keep model loaded** enabled.
Non-default RoPE scaling, sliding-window attention, and unsupported hybrid layouts
are rejected rather than silently ignored. Qwen3.5 uses its own validated hybrid path.

Tokenizer files are loaded locally; no tokenizer is downloaded automatically.
Selection order is:

1. **Tokenizer directory**, if explicitly entered.
2. A neighboring `<encoder-stem>.tokenizer/` or `tokenizer/` directory.
3. This extension's `tokenizer/qwen3/`, `tokenizer/qwen3_vl/`, or
   `tokenizer/qwen2_5_vl/`, or `tokenizer/qwen3_5/` directory, according to the resolved family.
4. Matching tokenizer directories already shipped with Forge Neo: Flux.2 Klein
   for Qwen3, Krea2/Qwen-Image 2.1 for Qwen3-VL, Qwen-Image for Qwen2.5-VL,
   MOSS-Image-SFT for Qwen3.5.

Include `tokenizer_config.json` and either `tokenizer.json` or `vocab.json` plus
`merges.txt`, and any additional tokenizer assets from the same model.
Special token IDs and vocabulary bounds are checked. A supplied chat template is
used; Forge's encoding-only tokenizer configs fall back to standard text ChatML.
An explicit tokenizer override must still match the actual encoder.

## Safetensors formats and memory

- BF16 / FP16 / FP32 language weights.
- Plain or scaled FP8 E4M3 / E5M2, including mixed-precision files and
  `float8_e4m3fn_rowwise` / `float8_e5m2_rowwise` metadata (also `tensorwise`).
- Scalar, row, or explicit column scales; Comfy's `scale_weight` / `weight_scale`.
- HF block-wise FP8 with `weight_scale_inv`: fixed 128×128 blocks, including
  partial edge blocks. The stored value is the dequantization multiplier.
- Scaled INT8 and INT8 ConvRot with validated Hadamard group sizes.
- Neo W4A8 (`asym_w4a8_int8`): packed 4-bit weights, relative/group scales,
  channel scales, optional codebook/correction, and ConvRot metadata.
- Neo W6A8 (`w6a8_int8`): uniform 6-bit codes packed in low-nibble and high-bit
  planes, relative/group and channel scales, and ConvRot; 32-column groups by default.
- INT4 ConvRot (`convrot_w4a4`, often named `int4_convrot`): signed low-nibble-first
  codes, per-row scales, and the regular Hadamard rotation. The native path also
  quantizes activations to INT4; CPU/dense fallback retains full-precision activations.
- NVFP4 (`nvfp4`): packed E2M1 codes, swizzled E4M3 block scales and a global
  scale (`weight_scale_2`). UINT8-stored block-scale bytes are reinterpreted as FP8.
  Original padded dimensions can be declared by `orig_shape` in layer metadata.
  Nontrivial SmoothQuant `pre_quant_scale` is rejected rather than ignored.
- Per-layer `.comfy_quant` metadata and header `_quantization_metadata` are read.
  Malformed metadata, missing INT8 scales, and unsupported layouts are rejected.

With **Keep supported quantized weights compressed** enabled, row/tensor-scaled INT8 and FP8 linear
weights remain one byte per value. ConvRot rotates activations instead of expanding
weights. A self-checked Triton kernel accelerates eligible INT8 and row/tensor-scaled FP8 single-token
steps; PyTorch is the fallback. W4A8 keeps its original packed representation and
uses Comfy-Kitchen's backend, with a dense per-layer fallback if no kernel applies.
The status reports compressed layer counts, W4A8/GGUF counts, and INT8 conversion
counts. GEMV verification is separate for INT8, FP8 E4M3, and FP8 E5M2;
status reports only storage types present in the current model. This fused
single-token kernel reads compressed weights directly and does not use FP8
Tensor Core matrix multiplication. Prefill and unsupported devices use PyTorch.
W4A8/W6A8 and INT4 ConvRot use Comfy-Kitchen's selected backend. NVFP4 uses
native FP4 matrix multiplication only on eligible Blackwell GPUs (SM 10.0+),
with dynamic activation scaling; on earlier CUDA GPUs, it keeps packed
weights resident and dequantizes only the current layer into the compute dtype.
This saves resident memory but does not promise native FP4 speed. GGUF uses Neo's
per-layer decode operations. Model details separate W4A8, W6A8, INT4 ConvRot,
NVFP4 and GGUF counts and report the packed INT4/NVFP4 execution path.

Block/column-scaled FP8 or INT8 linear weights that do not fit the resident
row-scale layout are decoded and converted to row-wise INT8 when compression is
on. This is another lossy quantization step. Disable **Keep supported quantized weights compressed**
to decode into the chosen compute dtype without that conversion.
**Also compress bf16 / fp16 weights to int8** additionally converts dense linear
weights on load and only applies when **Keep supported quantized weights compressed** is enabled.
Embeddings and normalization weights stay dense, so total memory is larger than
packed linear weights alone. Models must fit on the selected device; automatic
CPU/GPU layer offloading is not implemented.

MXFP8 and unrelated INT4/AWQ/GPTQ layouts are not supported by this version.
A filename containing `int4` is not enough: `convrot_w4a4` is a specific layout,
not a generic decoder for every 4-bit checkpoint. A file's integer tensor dtype alone is not sufficient
to identify its quantization.

## GGUF

Single-file GGUF encoders with architectures `qwen3`, `qwen3vl`, `qwen2vl`
(Qwen2.5-VL 7B), and `qwen35` (dense Qwen3.5 9B) are accepted. Language tensor names are mapped to the corresponding
HF backbone; unknown language tensors are rejected. Use the matching local family
tokenizer as described above. Embedded GGUF tokenizer metadata is not imported.
Split/sharded GGUF sets must first be merged into one GGUF file outside the extension.

Qwen3.5 reverses llama.cpp's tiled V-head ordering, zero-centered RMSNorm offset,
negative exponential decay values, and squeezed convolution dimensions. Both
separate and merged QKV/Z (`attn_qkv_gate`) and beta/alpha (`ssm_beta_alpha`)
projections are supported. Complete packed rows are sliced/permuted before baking;
output projection column permutations are applied to activations, keeping those
weights packed. Dense embedding/norm/recurrent parameters use CPU decoding.
Declared trailing MTP layers are skipped. Hadamard-folded Prism exports and MoE
architectures are rejected; they require additional transform support.

With compression enabled, linear weights stay packed using Neo's GGUF operations.
Only the active layer is dequantized for computation; the full model is not expanded
to BF16. Embeddings/norms are decoded on CPU during loading. Neo's baked linear
records use FP16 scales, even when the selected activation/compute dtype differs.
Resident record sizes can differ from file sizes. Each token still requires
dequantizing every active linear weight and running a dense PyTorch linear
operation; this is not a fused GGUF matrix-vector kernel. Different quantization
layouts have different decoding costs, so a smaller file need not generate faster.
Dense embeddings/output heads, temporary decoded layers, and the KV cache also
consume VRAM. If Windows spills GPU allocations into shared system memory,
generation can slow drastically even though loading succeeded. This path does not promise
llama.cpp performance and does not require llama.cpp or a separate process.

Regression-tested layouts are Q8_0, Q6_K, Q5_K, Q4_K, Q5_0, Q5_1, Q4_0, and Q4_1.
Q4_K_M/Q5_K_M files mix tensor types and are handled per tensor. F16/F32 and BF16
storage are also recognized. Other quantizations require support in Neo's GGUF
mapping and have not all been validated here; unsupported types fail explicitly.

## Validation coverage and limitations

Validation covers small full causal models for W6A8, INT4 ConvRot and NVFP4,
CPU/GPU linear paths, and Qwen3.5 GGUF logits/cached generation with both split
and merged projections in FP32/BF16/FP16 on CPU and CUDA. Error-handling tests
check that failed loading and generation release model/temporary tensors before
emptying CUDA cache, including chained exceptions and **Keep model loaded**.
An actual MiniMax-H3 INT4 ConvRot K projection was also
loaded and executed on CPU and RTX 4060. Full generation also passed with the
real `qwen3_vl_4b_nvfp4_full.safetensors` encoder on RTX 4060, using per-layer
NVFP4 decoding. Full production W6A8 encoders and Blackwell-native NVFP4 have
not been validated end to end.
Short full-model generation passed with `Qwen3.5-9B-Q4_K_M.gguf` using BF16 on
CPU. The full 9B GPU attempt exceeded available VRAM on the test card; the mixed
precision CUDA path is covered by the smaller hybrid regression models above.
A community MiniMax file can be a pruned encoder: the local file labeled 32B has
50 language layers, which is not the standard 64-layer Qwen3-VL 32B profile.
Such variants still need a matching explicit architecture config and original
output head; storage-format support does not establish language-generation quality.

Future formats should follow actual Neo encoder availability: AWQ W4A16 may be
useful for weight-only 4-bit inference; MXFP8 is mainly a newer-hardware path;
GPTQ/AWQ variants need explicit packing/zero-point handling. They are not enabled
by file extension or integer dtype alone.

## Memory reporting

**Model details** separates the loaded tensor footprint from physical GPU usage:

- **Loaded weights/buffers** counts the actual unique tensor storage, including
  compressed codes, scales, embeddings, and the output head. Tied weights and
  aliased tensor views are counted once. This is not the file size or total runtime memory.
- **CUDA / CPU** classifies where those tensors are allocated. On Windows, a CUDA
  tensor paged into shared RAM still appears as CUDA; this does not measure physical residency.
- **Physical VRAM in use** reports device-wide physical usage, which saturates at
  the card's capacity and can include other programs.
- **PyTorch CUDA allocated / peak this run / reserved** reports process-wide
  allocator statistics. The peak resets before model preparation for each GPU
  refinement and includes loading and inference temporaries, KV/recurrent caches,
  and any other active PyTorch allocations in Forge. Reserved memory includes
  allocator caching and is not an additional amount to add to allocated memory.

Thus allocations and model storage can reveal a requirement beyond physical VRAM
when Windows allows shared-memory spill. Neither metric measures exact shared-RAM
residency, driver allocations, or complete process RAM. Model storage excludes
inference caches/temporaries; the process peak is not an isolated model-only measurement.

## Missing LM heads

A diffusion encoder may omit `lm_head.weight`. Known Qwen3 0.6B / 1.7B / 4B and
Qwen3-VL 2B / 4B profiles tie this weight to embeddings and need no extra file.
Untied models, including Qwen2.5-VL 7B, Qwen3/VL 8B, and Qwen3.5 9B, need the exact original
output head. A model's hidden size alone is never used to infer weight tying.

A missing output head is an expected setup condition: refinement stops with a
message in **Status** and a short console line, without a traceback or clearing
the existing result. Open **LM head**, choose a **Known original model source**,
and click **Download lm_head**. The source choice fills the repository field;
custom repositories can still be entered manually as `owner/model`, not a web URL.
These suggestions apply to standard original checkpoints:

| Encoder | Original repository ID |
| --- | --- |
| Qwen3.5 9B / MOSS-Image | `Qwen/Qwen3.5-9B` |
| Qwen2.5-VL 7B / Qwen-Image | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen3-VL 8B / Qwen-Image 2.1 | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen3 8B / Flux.2 Klein 9B | `Qwen/Qwen3-8B` |

The dropdown also includes standard Qwen3 14B/32B and Qwen3-VL 32B sources.
For a fine-tune, use its exact original checkpoint instead; the source menu is
guidance, not automatic compatibility detection. A diffusion model repository may
omit the output head even though it contains the encoder. Absence of the head in
that repository does not by itself establish tied embeddings. Download/network
errors are displayed beside the download controls without a console traceback.

Only `lm_head.weight` is fetched through HTTP range requests,
then stored in `models/prompt_refiner/lm_heads/`, as row-wise INT8 by default.
The token field is only needed for gated repositories. This is an explicit optional
weight download; inference itself stays local.

Choose the head explicitly. **Auto** uses an exact filename matching
`<encoder-stem>.safetensors`, or identical original-source metadata when available.
It no longer chooses a head merely because tensor dimensions match. Explicit
selection or renaming a head to the encoder stem binds it to that encoder; ensure
it came from the same original model/fine-tune. A same-sized head from a different
checkpoint can produce incorrect text. An explicit head overrides an embedded one.

Custom head files may contain `lm_head.weight` and optional `lm_head.weight_scale`.
Configuration, tokenizer, encoder, or head file changes invalidate the loaded-model
cache on the next refinement.

## Uninstall and stored files

1. Stop Forge Neo.
2. Delete the extension folder, normally
   `<forge>/extensions/sd-webui-prompt-refiner/`. This also removes saved presets
   (`presets.json`, `presets.json.bak`, and `settings.json`) and any tokenizers stored in its `tokenizer/<family>/` folders.
   Back up these files first if you want to keep them.
3. Delete downloaded output heads from
   `<forge>/models/prompt_refiner/lm_heads/`. These files are stored outside the
   extension and are not removed when its folder is deleted. You can remove
   `<forge>/models/prompt_refiner/` entirely if it contains only this extension's files.
4. If you supplied tokenizer directories or LM head files elsewhere, remove those
   separately when no longer needed.
5. Start Forge Neo again.

Text encoders in `models/text_encoder/`, `models/text_encoders/`, or a custom
text encoder directory are shared with image generation. Delete individual files
only if you no longer need them in Forge. The extension does not install a
separate service or additional Python packages, so no package uninstall is needed.
