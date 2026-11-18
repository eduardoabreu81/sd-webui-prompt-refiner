# Prompt Refiner for SD WebUI Forge Neo

Rewrite and expand image prompts using a supported text encoder you already have
in Forge Neo. The extension adds a **Prompt Refiner** accordion to txt2img and
img2img and runs locally inside Forge, without LM Studio, Ollama, or a separate
language-model server.

This is a fork of [AMGarkin/sd-webui-prompt-refiner](https://github.com/AMGarkin/sd-webui-prompt-refiner)
that moves the refiner from its own tab into the generation tabs, refines the
tab's prompt in place, and keeps its options in Forge's Settings.

## Install

In Forge Neo, open **Extensions > Install from URL**, paste this repository's URL,
install, and restart Forge. You can also clone the repository into
`<forge>/extensions/sd-webui-prompt-refiner/`.

Use an up-to-date **Forge Neo** installation. The extension reuses Forge's Python
packages; it does not install or upgrade them.

## Quick start

1. Place a supported `.safetensors` or `.gguf` text encoder in
   `<forge>/models/text_encoder/`. Subfolders, `models/text_encoders/`, and Forge's
   extra text encoder directory are also searched. Refresh the list after adding files.
2. Write your idea in the txt2img or img2img prompt as usual.
3. Open the **Prompt Refiner** accordion, select the encoder and a system prompt
   preset, and click **Refine prompt**. The refined text appears live, then
   replaces the tab's prompt.
4. Not good? Click **Refine prompt** again for a new variation of the same original,
   or **Restore original** to put your prompt back. The original stays visible in
   the accordion. Editing the prompt makes the edited text the new original.

Extra-network tags such as `<lora:name:1>` are not sent to the language model.
They are put back in the refined prompt: tags at the start of your prompt stay at
the start, the others are added at the end.

All other options are in **Settings > Prompt Refiner**: system prompt presets,
sampling, and model settings. Applied changes take effect on the next refinement.
Presets are one text in which each preset starts with a `### Name` line; after
applying, click 🔄 in the accordion to update the preset menu. The encoder and
preset choices are remembered with Forge's settings. Your prompts and results are
not saved by the extension. **Disable thinking via chat template** is checked by
default; dedicated thinking-only models cannot reliably disable reasoning.

## Choosing an encoder for speed

The refiner only writes text, so **it does not need the same encoder as your
image model**: a Qwen3-VL 4B encoder can write prompts for MOSS-Image, Anima, or
any other model. For speed, prefer **Qwen3-VL 4B** or **Qwen3 4B** stored as
**INT8 / INT8 ConvRot or scaled FP8** on an NVIDIA GPU, and enable **Keep the
refiner model loaded between runs** in Settings if you have spare VRAM.

Avoid larger or slower paths unless you need them: **Qwen3.5 9B** runs on the
extension's own PyTorch decoder, **W4A8/W6A8** and **GGUF** can decompress every
layer for each token when no fast kernel applies, and an encoder that does not fit
in VRAM spills into shared system memory on Windows. Any of these can drop output
to seconds or even minutes per token. Without an NVIDIA GPU, refinement runs on
the CPU. **Model details** shows the device, execution path, and memory use.

## Supported encoders

| Encoder family | Common Forge Neo use |
| --- | --- |
| Qwen3 | Z-Image, Flux.2 Klein, Anima |
| Qwen3-VL | Krea2, Qwen-Image 2.1 |
| Qwen2.5-VL 7B | Qwen-Image / Qwen-Image-Edit |
| Qwen3.5 9B (dense) | MOSS-Image |

Supported layouts include BF16/FP16/FP32, scaled FP8, INT8/ConvRot, W4A8/W6A8,
INT4 ConvRot, NVFP4, and selected GGUF quantizations. Compatibility depends on the
actual model and weight layout. See the [technical reference](docs/TECHNICAL.md)
for exact formats, setup requirements, and testing coverage. MoE, CLIP, T5/UMT5,
and other model families are not supported.

## Sampling recommendations

The default targets **Qwen3-VL 4B Instruct / Krea 2**: preserve the idea's explicit
subjects, actions, colors, and other constraints while adding compatible details.
Its temperature, top-p, top-k, repetition penalty, and **presence penalty 1.5**
follow the model card's [VL recommendations](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct#vl).
Min-p starts disabled. These are starting points, not a
quality guarantee; use the natural-language system preset and review the result.

| Model / use | Temperature | Top-p | Top-k | Min-p | Repetition penalty | Presence penalty |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-VL 4B Instruct: balanced expansion (default)** | **0.7** | **0.8** | **20** | **0** | **1.0** | **1.5** |
| Qwen3-VL Instruct: optional broader text profile | 1.0 | 1.0 | 40 | 0 | 1.0 | 2.0 |
| Qwen3: thinking disabled | 0.7 | 0.8 | 20 | 0 | 1.0 | 0 |
| Qwen3: thinking enabled | 0.6 | 0.95 | 20 | 0 | 1.0 | 0 |
| Qwen3.5 9B: thinking disabled | 0.7 | 0.8 | 20 | 0 | 1.0 | 1.5 |
| Qwen3.5 9B: thinking enabled | 1.0 | 0.95 | 20 | 0 | 1.0 | 1.5 |
| Qwen2.5-VL 7B Instruct: conservative rewrite | 0 (greedy) | ignored | ignored | ignored | 1.05 | 0 |

The optional Qwen3-VL text profile and Qwen3/Qwen3.5 modes follow their model-card
recommendations: [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct#text),
[Qwen3](https://huggingface.co/Qwen/Qwen3-4B#best-practices), and
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices).
The Qwen2.5-VL row uses greedy decoding as a practical equivalent to its original
[near-zero temperature](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/generation_config.json).
The broader text profile is an alternative to try, not the default for preserving an idea.

For the default profile, try temperature **0.4-0.6** for a more restrained rewrite
or **0.8-0.9** for more variety. Leave the other values fixed while comparing.
Presence penalty discourages reusing tokens already generated; it excludes your
idea and system instructions and is applied once per token, not per occurrence.
The default **1.5** discourages repetition, including repeated endings in longer
descriptions such as MOSS prompts. If the output becomes too varied or avoids
necessary wording, lower it to **0.2-0.5**, or use **0** to disable it. It does not
guarantee that every repetition loop is prevented.
Repetition penalty above **1.0** also penalizes input tokens; try **1.03-1.05** only
if needed. Neither penalty guarantees that the model preserves the input's meaning.

**Temperature 0** selects greedy decoding and ignores top-p/top-k/min-p; both
penalties still apply. Keep **Seed -1** for variation, or use a fixed seed when
comparing settings. **Disable thinking** remains checked by default. Thinking
profiles need that checkbox cleared and may need a higher **Max new tokens** limit.

Sampling values are set in **Settings > Prompt Refiner > Sampling**; the table's
first row is the default. Profiles are set manually: selecting an encoder does not
replace your choices or automatically import a `generation_config.json`.

## What to expect

- **Results depend on the encoder and instructions.** Instruct models usually suit
  rewriting better than base models. Review the result before generating an image;
  the encoder can add details or change the meaning of your idea.
- **Some encoders need an output head.** If Forge's encoder omits the weights needed
  to generate text, the status message explains what is missing. Expand **Download a
  missing LM head** in the accordion, select a known original model source, and
  download its matching head; it becomes the LM head selected in Settings. This optional
  download is stored locally; do not substitute a head from a different model.
- **Speed and memory depend on your hardware and model.** The GPU path targets
  NVIDIA CUDA across different VRAM capacities. Choose an encoder that fits with
  room for inference; a small file does not guarantee low memory use or faster output.
  Quantization can reduce memory, but fast kernels vary by GPU and installed backend.
  CPU inference is available. AMD/ROCm is untested.
- **Models unload after refinement by default.** Enable **Keep the refiner model
  loaded between runs** in Settings for repeated use if you have enough memory. **Model details** shows diagnostics and
  provides a manual unload button. Failed runs always release the refiner's model
  and temporary tensors, even when the model is kept loaded.
  Increase **Max new tokens** if output is cut short.

For tokenizer overrides, missing heads, memory troubleshooting, or choosing a
quantization, read the [technical reference](docs/TECHNICAL.md).
An unknown model signature is reported in **Status** and a short console message
without a traceback; supply the matching original HF config as described there.

## Uninstall

1. Stop Forge Neo and delete `<forge>/extensions/sd-webui-prompt-refiner/`.
   This removes tokenizers stored inside the extension. Options and presets stay
   in Forge's `config.json` as `prompt_refiner_*` entries; they are unused once the
   extension is removed. Copy your presets text from Settings first if needed.
2. Delete downloaded LM heads in `<forge>/models/prompt_refiner/lm_heads/`.
   These are outside the extension folder and must be removed separately.
3. Remove any tokenizer or head files you placed elsewhere if no longer needed,
   then restart Forge.

Keep shared text encoders in Forge's model directories if you still use them for
image generation. No separate service or Python package uninstall is needed.
