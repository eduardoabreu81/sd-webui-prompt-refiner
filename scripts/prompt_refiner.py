"""
Prompt Refiner - extension for Forge Neo

Adds a "Prompt Refiner" accordion to txt2img and img2img that rewrites / expands
the tab's prompt with a local LLM.
The LLM is the very same text encoder file you already use for image generation
(e.g. qwen3vl_4b_fp8_scaled.safetensors from models/text_encoder).
No Ollama / LM Studio / external server is needed.

Supported: text-only Qwen3, Qwen3-VL, Qwen2.5-VL, and dense Qwen3.5 9B.
Safetensors and GGUF are accepted; Qwen3.5 supports safetensors and standard llama.cpp GGUF.
Compressed layouts include scaled FP8, INT8, INT8 ConvRot, and Neo W4A8.
MoE, other model families, and image inputs are not supported.
"""

import gc
import html
import json
import queue
import random
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import torch

from modules import script_callbacks, scripts, shared
from modules.options import categories
from modules.paths import models_path, script_path

EXT_DIR = Path(__file__).resolve().parents[1]
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))

# Forge imports extension scripts without guaranteeing their root is on sys.path.
from prompt_refiner_lib import checkpoints, models  # noqa: E402
from prompt_refiner_lib.sampling import SAMPLING_DEFAULTS, sampling_kwargs  # noqa: E402
from prompt_refiner_lib.lm_heads import KNOWN_HEAD_SOURCES, MissingLMHeadError  # noqa: E402
from prompt_refiner_lib.quantization import PackedLinear, QuantLinear, _FP8_TRITON, _TRITON  # noqa: E402

PRESETS_FILE = EXT_DIR / "presets.json"
LOCAL_TOKENIZER_DIR = EXT_DIR / "tokenizer"

_RUN_LOCK = threading.Lock()
_STOP = threading.Event()

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer
except Exception:  # keeps the tab loadable even if transformers changes its layout
    StoppingCriteria = StoppingCriteriaList = BaseStreamer = object

# --------------------------------------------------------------------------------------
# System prompt presets
# --------------------------------------------------------------------------------------

DEFAULT_PRESETS = {
    "Natural language (Krea / Flux / Qwen-Image / Z-Image)": (
        "You are an expert prompt writer for modern text-to-image models that understand natural language.\n"
        "Rewrite the user's input into one rich, vivid, well-structured description.\n"
        "Rules:\n"
        "- Start with the main subject and what it is doing, then describe environment, composition, lighting, "
        "materials/textures, colors, mood, and camera/lens details where it makes sense.\n"
        "- Keep every detail the user gave; never contradict it. Invent only fitting details.\n"
        "- Use flowing prose, no bullet points, no headings, no negations ('no', 'without').\n"
        "- If the user wants text in the image, keep it in \"quotation marks\" and say where it is placed.\n"
        "- Write 2-3 paragraphs at most.\n"
        "Respond ONLY with the final prompt. No introduction, no explanation, no quotes around it."
    ),
    "Short idea -> detailed prompt": (
        "You turn a short idea into a detailed image generation prompt.\n"
        "Expand the idea with a concrete subject, setting, composition, lighting, style and mood. "
        "Stay faithful to the idea. Output a single paragraph of plain prose.\n"
        "Respond ONLY with the final prompt."
    ),
    "Danbooru tags (SDXL / Anima / anime)": (
        "You convert and improve prompts for anime-style image models that use Danbooru tags.\n"
        "Output a single comma-separated line of lowercase tags: subject count, character, "
        "appearance, clothing, pose, expression, setting, lighting, composition, quality tags.\n"
        "Keep all concepts from the user's input. Do not write sentences.\n"
        "Respond ONLY with the tag line."
    ),
    "Comma-separated keywords (SD 1.5 / SDXL)": (
        "You improve prompts for Stable Diffusion 1.5 / SDXL.\n"
        "Output one line of comma-separated keywords and short phrases: subject, details, setting, "
        "lighting, camera, style, quality boosters. Keep everything from the user's input.\n"
        "Respond ONLY with the prompt line."
    ),
    "Light polish (keep wording)": (
        "You lightly polish an image generation prompt: fix grammar and spelling, remove redundancy, "
        "and add at most a few precise visual details. Keep the user's wording, order and language style "
        "as much as possible.\n"
        "Respond ONLY with the final prompt."
    ),
}


# Presets live in Settings > Prompt Refiner as one text: each preset starts with a
# "### Name" line followed by its instructions. A presets.json from an earlier
# version only seeds the default text.
PRESET_HEADER = "### "


def presets_to_text(presets):
    return "\n\n".join(f"{PRESET_HEADER}{name}\n{text.strip()}" for name, text in presets.items())


def presets_from_text(text):
    """Parse "### Name" sections; text before the first header is ignored."""
    presets, name, lines = {}, None, []
    for line in (text or "").splitlines():
        title = line[len(PRESET_HEADER):].strip() if line.startswith(PRESET_HEADER) else ""
        if title:
            if name is not None:
                presets[name] = "\n".join(lines).strip()
            name, lines = title, []
        else:
            lines.append(line)
    if name is not None:
        presets[name] = "\n".join(lines).strip()
    return presets


def _default_presets_text():
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data and all(
            isinstance(name, str) and name.strip() and isinstance(text, str) for name, text in data.items()
        ):
            return presets_to_text(data)
    except (OSError, ValueError):
        pass
    return presets_to_text(DEFAULT_PRESETS)


def current_presets():
    """Presets from Settings; a text without any "### Name" header falls back to the defaults."""
    return presets_from_text(_opt("presets")) or dict(DEFAULT_PRESETS)


# --------------------------------------------------------------------------------------
# Finding text encoders and tokenizer
# --------------------------------------------------------------------------------------

_ENCODER_PATHS = {}
_SIGNATURE_CACHE = {}


def _encoder_dirs():
    dirs = [Path(models_path) / "text_encoder", Path(models_path) / "text_encoders"]
    extra = getattr(shared.cmd_opts, "text_encoder_dir", None)
    if extra:
        dirs.append(Path(extra))
    return [d for d in dirs if d.is_dir()]


def _is_supported_file(path):
    """Cache architecture header checks until the checkpoint changes on disk."""
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False
    if key not in _SIGNATURE_CACHE:
        _SIGNATURE_CACHE[key] = checkpoints.is_supported(path)
    return _SIGNATURE_CACHE[key]


def list_encoders():
    """Discover supported safetensors/GGUF files and disambiguate duplicate names."""
    _ENCODER_PATHS.clear()
    seen = set()
    for directory in _encoder_dirs():
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in (".safetensors", ".gguf") or not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen or not _is_supported_file(path):
                continue
            seen.add(resolved)
            name = str(path.relative_to(directory))
            if name in _ENCODER_PATHS:
                name = str(path)
            _ENCODER_PATHS[name] = path
    return list(_ENCODER_PATHS)


# --------------------------------------------------------------------------------------
# Loading supported Qwen language weights into a text-only causal model
# --------------------------------------------------------------------------------------

def _pick_dtype(name: str, device: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    # auto
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.bfloat16


# --------------------------------------------------------------------------------------
# Separate "LM head" files for encoders that were shipped without lm_head.weight
# --------------------------------------------------------------------------------------
# Diffusion encoders may omit the language output projection. Model profiles
# determine whether embeddings are genuinely tied; untied models need exact weights.
LM_HEAD_DIR = Path(models_path) / "prompt_refiner" / "lm_heads"
_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def list_lm_heads():
    try:
        LM_HEAD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    return sorted(p.name for p in LM_HEAD_DIR.glob("*.safetensors"))


def _sidecar_shape(path):
    from safetensors import safe_open

    with safe_open(str(path), "pt", device="cpu") as f:
        return list(f.get_slice("lm_head.weight").get_shape())


def _pick_lm_head(choice, enc_path, vocab, hidden, profile=None):
    """Select explicitly, by exact encoder stem, or by matching original source metadata.

    Equal dimensions do not establish compatibility: unrelated fine-tunes often
    share the same shape. An explicit selection is the user's checkpoint binding.
    """
    from safetensors import safe_open

    def validate(path):
        if not path.is_file():
            raise RuntimeError(f"LM head file not found: {path}")
        if _sidecar_shape(path) != [vocab, hidden]:
            raise RuntimeError(f"LM head {path.name} does not match vocabulary/hidden size {[vocab, hidden]}.")
        return path

    if choice and choice != "Auto":
        path = LM_HEAD_DIR / choice
        if path.resolve().parent != LM_HEAD_DIR.resolve():
            raise RuntimeError("LM head selection must be a file in the LM head directory.")
        return validate(path)
    exact = LM_HEAD_DIR / f"{enc_path.stem}.safetensors"
    if exact.is_file():
        return validate(exact)
    source = (profile or {}).get("source", "")
    matches = []
    if source:
        for path in sorted(LM_HEAD_DIR.glob("*.safetensors")):
            with safe_open(str(path), "pt", device="cpu") as f:
                if (f.metadata() or {}).get("source") == source:
                    matches.append(validate(path))
    if len(matches) > 1:
        raise RuntimeError("Several LM heads match the original source; select one explicitly.")
    return matches[0] if matches else None


def _fetch_range(url, headers, a, b):
    """Generator: download bytes a..b (inclusive) with an HTTP Range request, yielding (done, total). Returns the bytes."""
    import requests

    total = b - a + 1
    r = requests.get(url, headers={**headers, "Range": f"bytes={a}-{b}"}, stream=True, timeout=60)
    try:
        if r.status_code != 206:
            raise RuntimeError(f"The server did not honour the range request (HTTP {r.status_code}), "
                               "so only the lm_head cannot be downloaded from this repo.")
        buf = bytearray()
        for chunk in r.iter_content(1 << 22):
            buf += chunk
            yield len(buf), total
        if len(buf) != total:
            raise RuntimeError(f"Incomplete download ({len(buf)} of {total} bytes).")
        return buf
    finally:
        r.close()


def _create_lm_head(repo_id, as_int8, token):
    """Generator of (html, dropdown_update). Downloads ONLY lm_head.weight from a Hugging Face repo."""
    import struct

    from huggingface_hub import hf_hub_download, hf_hub_url
    from safetensors.torch import save_file

    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise RuntimeError("Enter a Hugging Face repo id, e.g. Qwen/Qwen3.5-9B")
    token = (token or "").strip() or None
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    yield _bar(None, "Looking up lm_head.weight ...", busy=True), gr.update()
    key = None
    fname = "model.safetensors"
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    try:
        idx = hf_hub_download(repo_id, "model.safetensors.index.json", token=token)
    except EntryNotFoundError:
        idx = None  # single-file repo; access, gated-repo and network errors are reported as they are
    if idx:
        wm = json.loads(Path(idx).read_text(encoding="utf-8"))["weight_map"]
        key = next((k for k in wm if k.endswith("lm_head.weight")), None)
        if key is None:
            raise RuntimeError("This repository has no lm_head.weight. Diffusion encoders may omit it: choose the original language model repository. Models with tied embeddings need no separate head.")
        fname = wm[key]
    url = hf_hub_url(repo_id, fname)

    # header: 8 bytes length + JSON
    head = yield from _drain(_fetch_range(url, headers, 0, 7), "Reading header")
    n = struct.unpack("<Q", bytes(head))[0]
    header = json.loads(bytes((yield from _drain(_fetch_range(url, headers, 8, 8 + n - 1), "Reading header"))))
    key = key or next((k for k in header if k.endswith("lm_head.weight")), None)
    if key is None or key not in header:
        raise RuntimeError(f"lm_head.weight not found in {fname} of {repo_id}.")
    info = header[key]
    if info["dtype"] not in _ST_DTYPES:
        raise RuntimeError(f"Unexpected lm_head dtype {info['dtype']}.")
    shape = info["shape"]
    b0, b1 = info["data_offsets"]
    base = 8 + n

    data = yield from _drain(_fetch_range(url, headers, base + b0, base + b1 - 1), "Downloading lm_head")
    w = torch.frombuffer(data, dtype=_ST_DTYPES[info["dtype"]]).reshape(shape)

    yield _bar(None, "Converting and saving ...", busy=True), gr.update()
    LM_HEAD_DIR.mkdir(parents=True, exist_ok=True)
    out_name = repo_id.replace("/", "__") + (".int8" if as_int8 else ".bf16") + ".safetensors"
    out = LM_HEAD_DIR / out_name
    if as_int8:
        q = torch.empty(shape, dtype=torch.int8)
        sc = torch.empty(shape[0], dtype=torch.float32)
        for i in range(0, shape[0], 8192):  # chunked to keep RAM low
            blk = w[i:i + 8192].float()
            c = (blk.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
            q[i:i + 8192] = torch.round(blk / c).clamp(-128, 127).to(torch.int8)
            sc[i:i + 8192] = c.squeeze(1)
        tensors = {"lm_head.weight": q, "lm_head.weight_scale": sc.view(-1, 1).contiguous()}
    else:
        tensors = {"lm_head.weight": w.contiguous()}
    save_file(tensors, str(out), metadata={"source": repo_id, "format": "int8-rowwise" if as_int8 else "dense"})
    mb = out.stat().st_size / 1e6
    names = list_lm_heads()
    yield (_bar(1.0, f"Saved {out_name} ({mb:.0f} MB). It is now the selected LM head in Settings > Prompt Refiner."),
           gr.update(choices=["Auto"] + names, value=out_name))


def _drain(gen, label):
    """Run a _fetch_range generator, yielding progress bars for the UI, and return its bytes."""
    while True:
        try:
            done, total = next(gen)
        except StopIteration as stop:
            return stop.value
        yield _bar(done / total, f"{label}: {done / 1e6:.0f} / {total / 1e6:.0f} MB"), gr.update()


def create_lm_head_ui(repo_id, token):
    """Report setup, network, or file errors beside the download controls, without raising.

    A completed download becomes the selected LM head in Settings.
    """
    try:
        for bar, update in _create_lm_head(repo_id, _opt("lm_head_int8"), token):
            if isinstance(update, dict) and update.get("value"):
                _set_option("lm_head", update["value"])
            yield bar
    except Exception as exc:
        message = f"LM head download not completed: {exc}"
        print(f"[Prompt Refiner] {message}", flush=True)
        yield _bar(0, message)


def build_model(path, dtype, rope_choice, work_device="cpu", keep_quantized=True, compress=False,
                lm_head_choice="Auto", family="Auto"):
    """Build a text-only causal model using validated architecture and storage adapters."""
    return models.build_model(path, dtype, rope_choice, work_device, keep_quantized, compress,
                              lm_head_choice, family, _pick_lm_head)


# --------------------------------------------------------------------------------------
# Runtime state
# --------------------------------------------------------------------------------------

_STATE = {"key": None, "model": None, "tokenizer": None, "info": ""}


def unload_model():
    _STATE.update(key=None, model=None, tokenizer=None, info="")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model unloaded."


def _free_forge_vram():
    try:
        from backend import memory_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_model(encoder_name, device_choice, dtype_choice, rope_choice, free_vram, keep_q=True, compress=False,
               lm_choice="Auto", family="Auto", tokenizer_override=""):
    """Cache a model only while its weights, configuration, and tokenizer stay unchanged."""
    path = _ENCODER_PATHS.get(encoder_name)
    if path is None:
        list_encoders()
        path = _ENCODER_PATHS.get(encoder_name)
    if path is None:
        raise RuntimeError("Select a supported Qwen text encoder in models/text_encoder.")

    device = "cuda" if (device_choice.startswith("auto") and torch.cuda.is_available()) or device_choice == "cuda" else "cpu"
    dtype = _pick_dtype(dtype_choice, device)
    resolved = models.inspect_profile(path, family)
    tokenizer_dir = models.find_tokenizer(path, resolved, script_path, LOCAL_TOKENIZER_DIR, tokenizer_override)
    def stamp(p):
        return (str(p), p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else (str(p), None, None)
    watched = [path, path.with_suffix(".config.json"), path.parent / "config.json"]
    watched.extend(sorted(tokenizer_dir.glob("*")))
    watched.extend(sorted(LM_HEAD_DIR.glob("*.safetensors")))
    key = (tuple(stamp(p) for p in watched), device, str(dtype), rope_choice, bool(keep_q),
           bool(compress), lm_choice, family, str(tokenizer_dir))

    if _STATE["key"] == key and _STATE["model"] is not None:
        return _STATE["model"], _STATE["tokenizer"], device

    unload_model()
    if device == "cuda" and free_vram:
        _free_forge_vram()

    tokenizer = models.load_tokenizer(tokenizer_dir, resolved)
    model, info = build_model(path, dtype, rope_choice, device, keep_q, compress, lm_choice, family)
    info += f"; tokenizer: {tokenizer_dir}"
    model.to(device)  # no dtype argument on purpose: keeps rotary buffers in fp32

    _STATE.update(key=key, model=model, tokenizer=tokenizer, info=info)
    return model, tokenizer, device


def _build_chat(system, user, empty_think, tokenizer, family):
    """Use the selected tokenizer's chat template and limit thinking controls to switchable Qwen models."""
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user.strip()})
    kwargs = {"enable_thinking": False} if empty_think and family in ("Qwen3", "Qwen3.5") else {}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    if empty_think and family in ("Qwen3", "Qwen3.5") and not text.rstrip().endswith("</think>"):
        text += "<think>\n\n</think>\n\n"
    return text


def _clean_output(text: str) -> str:
    if "</think>" in text and "<think>" not in text:  # template already opened the block (Thinking-2507)
        text = "<think>" + text
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL)  # unfinished thinking
    stripped = stripped.strip()
    # Keep reasoning hidden even when the model has not produced a final prompt yet.
    stripped = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", stripped).strip()
    if len(stripped) > 1 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        stripped = stripped[1:-1].strip()
    return stripped


class _TokenStreamer(BaseStreamer):
    """Collects generated token ids so the UI thread can show live progress."""

    def __init__(self, q):
        self.q = q
        self.ids = []
        self._skip_prompt = True
        self.t_first = None
        self.t_warm = None   # time the 8th token arrived (Triton compile / autotune happens before this)
        self.n_warm = 0
        self.t_last = None

    def put(self, value):
        if self._skip_prompt:  # first call carries the prompt itself
            self._skip_prompt = False
            return
        if self.t_first is None:
            self.t_first = time.time()
        self.ids.extend(value.reshape(-1).tolist())
        self.t_last = time.time()
        if self.t_warm is None and len(self.ids) >= 8:
            self.t_warm, self.n_warm = self.t_last, len(self.ids)
        self.q.put("tok")

    def end(self):
        self.q.put("end")


class _StopFlag(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), _STOP.is_set(), dtype=torch.bool, device=input_ids.device)


def _bar(frac, text, busy=False):
    pct = 100 if frac is None else max(0, min(100, int(frac * 100)))
    anim = "animation:prpulse 1.1s ease-in-out infinite;" if busy else ""
    return (
        "<style>@keyframes prpulse{0%,100%{opacity:.35}50%{opacity:1}}</style>"
        f'<div style="margin:4px 0"><div style="font-size:0.9em;margin-bottom:3px">{html.escape(str(text))}</div>'
        '<div style="height:12px;border-radius:6px;overflow:hidden;background:var(--background-fill-secondary);'
        'border:1px solid var(--border-color-primary)">'
        f'<div style="height:100%;width:{pct}%;background:var(--color-accent);transition:width .2s;{anim}"></div>'
        "</div></div>"
    )


def _model_storage_bytes(model):
    """Count unique retained tensor storages, including packed buffers and tied aliases.

    This is the actual loaded weight/buffer footprint, not the checkpoint file size
    or a logical GGUF shape. It excludes runtime KV/recurrent state and temporaries.
    CUDA storage remains CUDA-tagged when Windows pages it into shared host memory.
    """
    totals = {}
    seen = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_meta:
            continue
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr())
        if identity in seen:
            continue
        seen.add(identity)
        totals[tensor.device.type] = totals.get(tensor.device.type, 0) + storage.nbytes()
    return totals


def _describe(model, device):
    params = list(model.parameters())
    p = params[0]
    s = f"device {p.device}, {str(p.dtype).replace('torch.', '')}, attention: {getattr(model.config, '_attn_implementation', '?')}"
    nq = getattr(model, "_pr_quant_layers", 0)
    if nq:
        s += (f", {getattr(model, '_pr_w4_layers', 0)} W4A8 layers, {getattr(model, '_pr_w6_layers', 0)} W6A8 layers, "
              f"{getattr(model, '_pr_int4_layers', 0)} INT4 ConvRot layers, {getattr(model, '_pr_nvfp4_layers', 0)} NVFP4 layers, "
              f"{getattr(model, '_pr_gguf_layers', 0)} GGUF layers")
        for kind in ("convrot_w4a4", "nvfp4"):
            paths = {getattr(module, "_last_path", "not used yet") for module in model.modules()
                     if isinstance(module, PackedLinear) and module.kind == kind}
            if paths:
                s += f", {kind}: {', '.join(sorted(paths))}"
        # Report only storage types actually present in this model, not a previous run.
        kinds = {module.weight.dtype for module in model.modules() if isinstance(module, QuantLinear)}
        for dtype in sorted(kinds, key=str):
            state = _TRITON if dtype == torch.int8 else _FP8_TRITON.get(dtype, {"state": None})
            kernel = {True: "Triton (verified)", False: "PyTorch fallback", None: "not verified / PyTorch fallback"}[state["state"]]
            s += f", {str(dtype).replace('torch.', '')} GEMV: {kernel}"
        if getattr(model, "_pr_gguf_layers", 0):
            s += ", GGUF: per-layer dequantization + PyTorch linear"
    storage = _model_storage_bytes(model)
    s += f", loaded weights/buffers {sum(storage.values()) / 1e9:.2f} GB"
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        s += (
            f" (CUDA {storage.get('cuda', 0) / 1e9:.2f} GB, CPU {storage.get('cpu', 0) / 1e9:.2f} GB)"
            f", GPU {torch.cuda.get_device_name(0)}, physical VRAM in use {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB"
            f", PyTorch CUDA allocated {allocated:.2f} GB, peak this run {peak:.2f} GB, reserved {reserved:.2f} GB (process totals)"
        )
        n_cpu = sum(1 for q in params if q.device.type == "cpu")
        if n_cpu:
            s += f" - WARNING: {n_cpu} weight tensors are still on the CPU"
    return s


def run_refine(
    encoder_name, prompt, system_prompt, temperature, top_p, top_k, min_p, rep_penalty,
    max_new_tokens, seed, device_choice, dtype_choice, rope_choice, keep_loaded, free_vram, empty_think,
    keep_q=True, compress=False, lm_choice="Auto", family="Auto", tokenizer_override="",
    presence_penalty=SAMPLING_DEFAULTS["presence_penalty"],
):
    """Generator: streams the result text, a status line and a progress bar while generating."""
    if not prompt or not prompt.strip():
        message = "Enter a prompt first."
        print(f"[Prompt Refiner] Skipped: {message}", flush=True)
        yield gr.update(), message, ""
        return
    if not encoder_name:
        message = "Select a supported Qwen safetensors or GGUF encoder first."
        print(f"[Prompt Refiner] Skipped: {message}", flush=True)
        yield gr.update(), message, ""
        return

    max_new_tokens = int(max_new_tokens)
    with _RUN_LOCK:
        _STOP.clear()
        print(f"[Prompt Refiner] Starting refinement: {encoder_name}", flush=True)
        t0 = time.time()
        model = tokenizer = enc = thread = None
        errors = []
        released = completed = False

        def release_run(exc=None):
            """Drop tensor owners before collecting garbage and returning CUDA cache.

            Exceptions retain failed loader/worker frames, including activations
            and partially loaded models. Clearing only the model cache cannot
            release those allocations while Gradio still owns the exception.
            """
            nonlocal model, tokenizer, enc, released
            if thread is not None and thread.is_alive():
                _STOP.set()
                thread.join()
            pending = [exc] if exc is not None else []
            pending.extend(errors)
            seen = set()
            while pending:
                error = pending.pop()
                if id(error) in seen:
                    continue
                seen.add(id(error))
                traceback.clear_frames(error.__traceback__)
                # Loader wrappers can retain an earlier allocation failure as
                # their cause/context even after their own frames are cleared.
                pending.extend(
                    linked for linked in (error.__cause__, error.__context__)
                    if linked is not None
                )
            errors.clear()
            model = tokenizer = enc = None
            unload_model()
            released = True

        try:
            # Reset before loading so the peak includes model construction and inference.
            if torch.cuda.is_available() and device_choice != "cpu":
                torch.cuda.reset_peak_memory_stats()
            cached = _STATE["model"] is not None
            msg = "Preparing selected model..." if cached else "Loading model from file (reading and decoding weights)..."
            yield gr.update(), msg, _bar(None, msg, busy=True)

            model, tokenizer, device = _get_model(encoder_name, device_choice, dtype_choice, rope_choice, free_vram, keep_q, compress, lm_choice, family, tokenizer_override)
            t_load = time.time() - t0

            thinking_only = bool(re.search(r"thinking[-_ ]?2507", encoder_name, re.IGNORECASE))
            thinking_note = ""
            if thinking_only and empty_think:
                thinking_note = "This Thinking-2507 checkpoint does not support disabling thinking; the checkbox was ignored. "
                print(f"[Prompt Refiner] {thinking_note.strip()}", flush=True)
            text = _build_chat(system_prompt, prompt, empty_think and not thinking_only, tokenizer, model._pr_profile["family"])
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            n_prompt = enc["input_ids"].shape[1]
            if n_prompt + max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Prompt plus requested output exceeds the model context window. Shorten the prompt or lower Max new tokens.")

            seed = int(seed)
            if seed < 0:
                seed = random.randint(0, 2**31 - 1)
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            eos_ids = [i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                   tokenizer.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0]
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids or None,
                pad_token_id=eos_ids[-1] if eos_ids else 0,
            )
            gen_kwargs.update(sampling_kwargs(
                temperature, top_p, top_k, min_p, rep_penalty,
                presence_penalty, n_prompt,
            ))

            q = queue.Queue()
            streamer = _TokenStreamer(q)
            gen_kwargs["streamer"] = streamer
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopFlag()])
            def worker():
                try:
                    with torch.inference_mode():
                        model.generate(**enc, **gen_kwargs)
                except BaseException as e:  # reported in the UI thread
                    errors.append(e)
                finally:
                    q.put("end")

            t1 = time.time()
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            def snapshot(final=False):
                ids = list(streamer.ids)
                n = len(ids)
                rate = (n - 1) / (time.time() - streamer.t_first) if streamer.t_first and n > 1 else 0.0
                shown = _clean_output(tokenizer.decode(ids, skip_special_tokens=True)) if ids else ""
                if streamer.t_first is None:
                    label = f"Reading prompt ({n_prompt} tokens)..."
                else:
                    label = f"Generating... {n}/{max_new_tokens} tokens - {rate:.1f} tok/s"
                return shown, n, rate, label

            last = 0.0
            finished = False
            while not finished:
                try:
                    finished = q.get(timeout=0.2) == "end"
                except queue.Empty:
                    pass
                now = time.time()
                if not finished and now - last >= 0.25:
                    shown, n, rate, label = snapshot()
                    yield shown, label, _bar(n / max_new_tokens if n else None, label, busy=(n == 0))
                    last = now

            thread.join()
            if errors:
                raise errors[0]

            result, n, rate, _ = snapshot(final=True)
            t_gen = time.time() - t1
            t_first = (streamer.t_first - t1) if streamer.t_first else t_gen
            stopped = " (stopped by user)" if _STOP.is_set() else ""
            steady = ""
            if streamer.t_warm and n - streamer.n_warm >= 8 and streamer.t_last > streamer.t_warm:
                sr = (n - streamer.n_warm) / (streamer.t_last - streamer.t_warm)
                steady = (f" Steady-state: {sr:.1f} tok/s (after warm-up; the first 8 tokens took "
                          f"{streamer.t_warm - streamer.t_first:.1f}s).")
            limit_note = ""
            if n >= max_new_tokens and not _STOP.is_set():
                limit_note = " Token limit reached - raise 'Max new tokens' (thinking models often need 2000+)."
            status = (f"{thinking_note}Done{stopped}.{limit_note}{steady} {n} tokens in {t_gen:.1f}s ({rate:.1f} tok/s), prompt processing {t_first:.1f}s, "
                      f"model load {t_load:.1f}s, seed {seed}.\n{_STATE['info']}; {_describe(model, device)}")
            print(f"[Prompt Refiner] Done{stopped}: {n} tokens in {t_gen:.1f}s ({rate:.1f} tok/s), seed {seed}.", flush=True)
            completed = True
            yield result, status, _bar(1.0, f"Done{stopped} - {n} tokens, {rate:.1f} tok/s")
        except (MissingLMHeadError, models.UnknownModelSignatureError) as exc:
            # Known setup failures belong in Status, not Gradio's exception queue.
            message = str(exc)
            print(f"[Prompt Refiner] Setup needed: {message}", flush=True)
            release_run(exc)
            yield gr.update(), message, ""
        except gr.Error as exc:
            release_run(exc)
            raise
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            print(f"[Prompt Refiner] Failed: {message}", flush=True)
            release_run(e)
            raise gr.Error(message) from None
        finally:
            if not released and (not keep_loaded or not completed):
                release_run()


# --------------------------------------------------------------------------------------
# UI: an accordion in txt2img and img2img
# --------------------------------------------------------------------------------------

# The refiner reads and replaces the tab's own prompt. The prompt before the last
# refinement is kept in the "original" field (the last event input) so it can be
# restored. Refining an unedited result again starts over from that original, so
# repeated clicks give new variations instead of refining a refinement; editing
# the prompt makes the edited text the new original.
# Per-tab browser state remembers the prompt text the refiner last wrote.
_PROMPT_JS_STATE = r"""
    const ta = gradioApp().querySelector('#TAB_prompt textarea');
    const states = window.promptRefinerState = window.promptRefinerState || {};
    const state = states['TAB'] = states['TAB'] || {};
"""

# Extra-network tags such as <lora:name:1> are not sent to the language model,
# which would rewrite or drop them.
READ_PROMPT_JS = r"""
(...args) => {
""" + _PROMPT_JS_STATE + r"""
    const current = ta ? ta.value : '';
    const last = args.length - 1;
    if (!(args[last] && current === state.applied)) { args[last] = current; }
    args[1] = args[last]
        .replace(/<[^<>]+>/g, ' ')
        .replace(/\s*,(\s*,)+/g, ',')
        .replace(/\s+/g, ' ')
        .replace(/^[\s,]+|[\s,]+$/g, '');
    return args;
}
"""

# The original's tags come back where they were: leading tags before the result,
# the others after it. Tags the result already contains are not repeated.
APPLY_JS = r"""
(text, original) => {
""" + _PROMPT_JS_STATE + r"""
    if (!ta || !text || !text.trim()) { return []; }
    const source = original || '';
    const lead = ((source.match(/^(?:\s*<[^<>]+>[\s,]*)+/) || [''])[0].match(/<[^<>]+>/g)) || [];
    const tags = [...new Set(source.match(/<[^<>]+>/g) || [])].filter((tag) => !text.includes(tag));
    ta.value = [...tags.filter((tag) => lead.includes(tag)), text.trim(),
                ...tags.filter((tag) => !lead.includes(tag))].join(' ');
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    state.applied = ta.value;
    return [];
}
"""

RESTORE_JS = r"""
(original) => {
""" + _PROMPT_JS_STATE + r"""
    if (ta && original) {
        ta.value = original;
        ta.dispatchEvent(new Event('input', {bubbles: true}));
        state.applied = undefined;
    }
    return [];
}
"""

# --------------------------------------------------------------------------------------
# Settings > Prompt Refiner: every option except the accordion's encoder and preset
# menus, saved in Forge's config.json and read when Refine is clicked.
# --------------------------------------------------------------------------------------

OPTION_PREFIX = "prompt_refiner_"
_CATEGORY = "prompt_refiner"
_HIDDEN = (None, "Prompt Refiner")  # saved like any option, not listed in Settings
DEVICE_CHOICES = ["auto (GPU if available)", "cuda", "cpu"]
DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]
ROPE_CHOICES = ["Auto (checkpoint configuration / model profile)", "1,000,000 (Qwen3)",
                "5,000,000 (Qwen3-VL)", "10,000,000 (Qwen3.5)"]


def _opt(key):
    return getattr(shared.opts, OPTION_PREFIX + key)


def _set_option(key, value):
    """Remember an accordion choice; a failed save only loses that memory."""
    try:
        shared.opts.set(OPTION_PREFIX + key, value)
        shared.opts.save(shared.config_filename)
    except Exception as exc:
        print(f"[Prompt Refiner] Could not save the {key} choice: {exc}", flush=True)


def on_ui_settings():
    categories.register_category(_CATEGORY, "Prompt Refiner")
    presets_section = ("prompt_refiner_presets", "Presets")
    sampling_section = ("prompt_refiner_sampling", "Sampling")
    model_section = ("prompt_refiner_model", "Model")

    def add(key, default, label, section, component=None, component_args=None, info=None, refresh=None):
        option = shared.OptionInfo(default, label, component, component_args, refresh=refresh,
                                   section=section, category_id=_CATEGORY)
        shared.opts.add_option(OPTION_PREFIX + key, option.info(info) if info else option)

    def slider(minimum, maximum, step):
        return {"minimum": minimum, "maximum": maximum, "step": step}

    add("encoder", "", "Text encoder", _HIDDEN)
    add("preset", "", "System prompt preset", _HIDDEN)

    add("presets", _default_presets_text(), "System prompt presets", presets_section, gr.Textbox, {"lines": 18},
        info='each preset starts with a "### Name" line followed by its instructions; '
             "after applying, click 🔄 in the accordion to update the preset menu")

    defaults = SAMPLING_DEFAULTS
    add("temperature", defaults["temperature"], "Temperature (0 = greedy)", sampling_section, gr.Slider, slider(0.0, 2.0, 0.05))
    add("top_p", defaults["top_p"], "Top-p", sampling_section, gr.Slider, slider(0.0, 1.0, 0.01))
    add("top_k", defaults["top_k"], "Top-k (0 = off)", sampling_section, gr.Slider, slider(0, 200, 1))
    add("min_p", defaults["min_p"], "Min-p (0 = off)", sampling_section, gr.Slider, slider(0.0, 1.0, 0.01))
    add("repetition_penalty", defaults["repetition_penalty"], "Repetition penalty", sampling_section, gr.Slider,
        slider(1.0, 2.0, 0.01), info="above 1 penalizes tokens in both the input and generated text")
    add("presence_penalty", defaults["presence_penalty"], "Presence penalty (0 = off)", sampling_section, gr.Slider,
        slider(0.0, 2.0, 0.05), info="discourages reusing generated tokens; the idea and instructions are excluded")
    add("seed", -1, "Seed (-1 = random)", sampling_section, gr.Number, {"precision": 0})
    add("max_new_tokens", 768, "Max new tokens", sampling_section, gr.Slider, slider(32, 4096, 16),
        info="generation stops earlier at the end of the answer; thinking models can need 2000+")
    add("disable_thinking", True, "Disable thinking via chat template (switchable Qwen3 / Qwen3.5 only)", sampling_section,
        info="dedicated Thinking-2507 checkpoints cannot disable thinking")

    add("keep_loaded", False, "Keep the refiner model loaded between runs", model_section,
        info="much faster repeated runs; keeps using VRAM while you generate images")
    add("free_vram", True, "Unload Forge's image models from VRAM before loading the refiner model", model_section)
    add("device", DEVICE_CHOICES[0], "Device", model_section, gr.Radio, {"choices": DEVICE_CHOICES})
    add("dtype", DTYPE_CHOICES[0], "Compute dtype", model_section, gr.Radio, {"choices": DTYPE_CHOICES})
    add("family", "Auto", "Model family", model_section, gr.Dropdown, {"choices": models.FAMILIES},
        info="override Auto for renamed encoders with ambiguous architecture")
    add("rope", ROPE_CHOICES[0], "RoPE theta", model_section, gr.Dropdown, {"choices": ROPE_CHOICES})
    add("tokenizer_directory", "", "Tokenizer directory (optional)", model_section,
        info="leave empty to use matching local Forge tokenizer files")
    add("keep_quantized", True, "Keep supported quantized weights compressed (recommended)", model_section)
    add("compress", False, "Also compress bf16 / fp16 weights to int8 on load", model_section,
        info="reduces weight memory; may affect quality")
    add("lm_head", "Auto", "LM head file", model_section, gr.Dropdown,
        lambda: {"choices": ["Auto"] + list_lm_heads()}, refresh=list_lm_heads,
        info="for untied encoders without output weights; download one from the accordion")
    add("lm_head_int8", True, "Store downloaded LM heads as int8 (half the size)", model_section)


script_callbacks.on_ui_settings(on_ui_settings)


def _run(encoder_name, idea, preset_name, source):
    """Refine with the options applied in Settings at click time, then report the
    text to apply: empty unless a complete, unstopped result was generated."""
    lm_head = _opt("lm_head")
    if lm_head != "Auto" and lm_head not in list_lm_heads():
        lm_head = "Auto"
    refine_args = (
        encoder_name, idea, current_presets().get(preset_name, ""),
        _opt("temperature"), _opt("top_p"), _opt("top_k"), _opt("min_p"), _opt("repetition_penalty"),
        _opt("max_new_tokens"), _opt("seed"), _opt("device"), _opt("dtype"), _opt("rope"),
        _opt("keep_loaded"), _opt("free_vram"), _opt("disable_thinking"), _opt("keep_quantized"), _opt("compress"),
        lm_head, _opt("family"), _opt("tokenizer_directory") or "", _opt("presence_penalty"),
    )
    text = summary = ""
    first = True
    for result, message, progress in run_refine(*refine_args):
        if isinstance(result, str):
            text = result
        # Keep routine status short and put model diagnostics in their own panel.
        summary, _, details = message.partition("\n")
        yield text, summary, progress, details, (source if first else gr.update()), ""
        first = False
    keep = gr.update()
    if _STOP.is_set():
        yield keep, f"{summary} The prompt was left unchanged.", keep, keep, keep, ""
    elif text and summary.startswith("Done"):
        yield keep, f"{summary} The prompt was replaced; Restore original brings it back.", keep, keep, keep, text


def _blocks(block):
    yield block
    for child in getattr(block, "children", None) or []:
        yield from _blocks(child)


def build_ui(tab):
    """Build the refiner inside an accordion of the txt2img or img2img tab."""
    encoders = list_encoders()
    presets = list(current_presets())
    saved_encoder, saved_preset = _opt("encoder"), _opt("preset")

    with gr.Accordion("Prompt Refiner", open=False, elem_id=f"prompt_refiner_{tab}_accordion") as accordion:
        with gr.Row():
            encoder = gr.Dropdown(
                label="Text encoder (models/text_encoder)", choices=encoders, scale=6,
                value=saved_encoder if saved_encoder in encoders else (encoders[0] if encoders else None),
            )
            preset = gr.Dropdown(
                label="System prompt preset", choices=presets, scale=4,
                value=saved_preset if saved_preset in presets else (presets[0] if presets else None),
            )
            refresh_btn = gr.Button("🔄", elem_classes=["tool"], scale=1, min_width=40)
        with gr.Row():
            run_btn = gr.Button("Refine prompt", variant="primary", scale=4)
            stop_btn = gr.Button("Stop", scale=1)
            restore_btn = gr.Button("Restore original", scale=2)
        output = gr.Textbox(label="Refined prompt (live)", lines=3, interactive=False, show_copy_button=True)
        original = gr.Textbox(
            label="Original prompt (before the last refinement)", lines=2, interactive=False,
            show_copy_button=True, placeholder="Refine prompt replaces this tab's prompt; the previous one is kept here.",
        )
        progress_html = gr.HTML(value="")
        status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Accordion("Model details", open=False):
            model_details = gr.Textbox(label="Loaded model details", interactive=False, lines=5)
            unload_btn = gr.Button("Unload model")

        with gr.Accordion("Download a missing LM head", open=False):
            gr.Markdown(
                "Known tied models need no extra weights. Qwen2.5-VL 7B, Qwen3.5 9B, and untied Qwen3/VL encoders "
                "without `lm_head.weight` need that exact tensor from the ORIGINAL model on Hugging Face "
                "(for MOSS-Image / Qwen3.5 9B: `Qwen/Qwen3.5-9B`). Enter it as `owner/model`; for fine-tunes, use "
                "their exact original checkpoint. The download is stored in `models/prompt_refiner/lm_heads/` and "
                "selected in **Settings > Prompt Refiner**. Matching tensor dimensions do not establish compatibility."
            )
            lm_source = gr.Dropdown(label="Known original model source", choices=list(KNOWN_HEAD_SOURCES),
                                    value=next(iter(KNOWN_HEAD_SOURCES)))
            with gr.Row():
                lm_repo = gr.Textbox(label="Hugging Face repo of the ORIGINAL model", placeholder="Qwen/Qwen3.5-9B", scale=4)
                lm_token = gr.Textbox(label="HF token (only for gated repos)", type="password", scale=2)
                lm_btn = gr.Button("Download lm_head", scale=1)
            lm_status = gr.HTML(value="")

        gr.Markdown("Sampling, model, and preset options: **Settings > Prompt Refiner**. They apply on the next refinement.")
        # Filled by READ_PROMPT_JS and by the generator; never shown. Only a
        # completed, unstopped result reaches apply_text and the prompt.
        prompt = gr.Textbox(visible=False)
        apply_text = gr.Textbox(visible=False)

    encoder.change(lambda name: _set_option("encoder", name or ""), inputs=[encoder], outputs=None, show_progress="hidden")
    preset.change(lambda name: _set_option("preset", name or ""), inputs=[preset], outputs=None, show_progress="hidden")

    def _refresh(current_encoder, current_preset):
        names, preset_names = list_encoders(), list(current_presets())
        return (
            gr.update(choices=names, value=current_encoder if current_encoder in names else (names[0] if names else None)),
            gr.update(choices=preset_names,
                      value=current_preset if current_preset in preset_names else (preset_names[0] if preset_names else None)),
        )

    refresh_btn.click(_refresh, inputs=[encoder, preset], outputs=[encoder, preset])
    lm_source.input(lambda name: KNOWN_HEAD_SOURCES.get(name, ""), inputs=[lm_source], outputs=[lm_repo])
    lm_repo.input(lambda repo: next((name for name, value in KNOWN_HEAD_SOURCES.items() if value == repo), next(iter(KNOWN_HEAD_SOURCES))),
                  inputs=[lm_repo], outputs=[lm_source])
    lm_btn.click(create_lm_head_ui, inputs=[lm_repo, lm_token], outputs=[lm_status], show_progress="hidden")

    run_btn.click(
        _run,
        inputs=[encoder, prompt, preset, original],
        outputs=[output, status, progress_html, model_details, original, apply_text],
        show_progress="hidden",
        js=READ_PROMPT_JS.replace("TAB", tab),
    ).success(fn=None, inputs=[apply_text, original], outputs=None, js=APPLY_JS.replace("TAB", tab))
    stop_btn.click(lambda: _STOP.set(), inputs=None, outputs=None)
    restore_btn.click(fn=None, inputs=[original], outputs=None, js=RESTORE_JS.replace("TAB", tab))
    unload_btn.click(lambda: (unload_model(), ""), outputs=[status, model_details])

    # Forge applies ui-config.json after the tab is built. The encoder and preset
    # menus are remembered in Forge's settings instead, and transient text must
    # not be restored, so every control in the accordion opts out.
    for component in _blocks(accordion):
        component.do_not_save_to_config = True
    return accordion


class PromptRefinerScript(scripts.Script):
    create_group = False

    def title(self):
        return "Prompt Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Refinement only edits the prompt box; it passes nothing to image generation.
        build_ui("img2img" if is_img2img else "txt2img")
        return []
