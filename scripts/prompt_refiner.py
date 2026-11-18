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
import os
import queue
import random
import re
import sys
import threading
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import torch

from modules import scripts, shared
from modules.paths import models_path, script_path

EXT_DIR = Path(__file__).resolve().parents[1]
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))

# Forge imports extension scripts without guaranteeing their root is on sys.path.
from prompt_refiner_lib import checkpoints, models  # noqa: E402
from prompt_refiner_lib.preferences import SettingsStore  # noqa: E402
from prompt_refiner_lib.sampling import SAMPLING_DEFAULTS, sampling_kwargs  # noqa: E402
from prompt_refiner_lib.lm_heads import KNOWN_HEAD_SOURCES, MissingLMHeadError  # noqa: E402
from prompt_refiner_lib.quantization import PackedLinear, QuantLinear, _FP8_TRITON, _TRITON  # noqa: E402

PRESETS_FILE = EXT_DIR / "presets.json"
SETTINGS_FILE = EXT_DIR / "settings.json"
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


_PRESET_LOCK = threading.RLock()


def load_presets():
    """Read presets without overwriting malformed files; missing files use defaults."""
    if not PRESETS_FILE.exists():
        return dict(DEFAULT_PRESETS)
    data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(
        not isinstance(name, str) or not name.strip() or not isinstance(text, str)
        for name, text in data.items()
    ):
        raise ValueError("The presets file must contain an object mapping names to text.")
    return data


def _replace_preset_file(path, content):
    """Replace a file atomically using a temporary file on the same filesystem."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_presets(data):
    """Keep one previous version before atomically replacing the current presets."""
    content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    with _PRESET_LOCK:
        if PRESETS_FILE.exists():
            _replace_preset_file(PRESETS_FILE.with_suffix(".json.bak"), PRESETS_FILE.read_bytes())
        _replace_preset_file(PRESETS_FILE, content)


def select_preset_ui(name):
    """Populate both editable fields from a user-selected preset without exceptions."""
    try:
        data = load_presets()
        return data.get(name, ""), name or "", ""
    except (OSError, ValueError) as exc:
        message = f"Could not read presets: {exc}. Check presets.json before trying again."
        print(f"[Prompt Refiner] {message}", flush=True)
        return gr.update(), gr.update(), message


def change_preset_ui(name, text=None, delete=False):
    """Save or delete a preset, preserving the editor on validation or disk errors."""
    name = (name or "").strip()
    if not name:
        message = "Select a preset to delete." if delete else "Enter a preset name before saving."
        print(f"[Prompt Refiner] {message}", flush=True)
        return gr.update(), gr.update(), gr.update(), message
    try:
        with _PRESET_LOCK:
            data = load_presets()
            existed = name in data
            if delete:
                if not existed:
                    return gr.update(), gr.update(), gr.update(), "The selected preset no longer exists."
                del data[name]
            else:
                data[name] = text or ""
            save_presets(data)
        selected = next(iter(data), None) if delete else name
        message = f'Preset "{name}" deleted.' if delete else f'Preset "{name}" {"updated" if existed else "saved"}.'
        print(f"[Prompt Refiner] {message}", flush=True)
        return gr.update(choices=list(data), value=selected), data.get(selected, ""), selected or "", message
    except (OSError, ValueError) as exc:
        message = f"Could not {'delete' if delete else 'save'} the preset: {exc}. Your edits are still in the editor; check presets.json and folder write permissions."
        print(f"[Prompt Refiner] {message}", flush=True)
        return gr.update(), gr.update(), gr.update(), message


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
    yield (_bar(1.0, f"Saved {out_name} ({mb:.0f} MB). It is selected below."),
           gr.update(choices=["Auto"] + names, value=out_name))


def _drain(gen, label):
    """Run a _fetch_range generator, yielding progress bars for the UI, and return its bytes."""
    while True:
        try:
            done, total = next(gen)
        except StopIteration as stop:
            return stop.value
        yield _bar(done / total, f"{label}: {done / 1e6:.0f} / {total / 1e6:.0f} MB"), gr.update()


def create_lm_head_ui(repo_id, as_int8, token):
    """Report setup, network, or file errors beside the download controls, without raising."""
    try:
        yield from _create_lm_head(repo_id, as_int8, token)
    except Exception as exc:
        message = f"LM head download not completed: {exc}"
        print(f"[Prompt Refiner] {message}", flush=True)
        yield _bar(0, message), gr.update()


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

# An empty idea field uses the tab's current prompt instead, without extra-network
# tags such as <lora:name:1>, which the language model would rewrite or drop.
FILL_IDEA_JS = r"""
(...args) => {
    if (!args[1] || !args[1].trim()) {
        const ta = gradioApp().querySelector('#TAB_prompt textarea');
        args[1] = (ta ? ta.value : '')
            .replace(/<[^<>]+>/g, ' ')
            .replace(/\s*,(\s*,)+/g, ',')
            .replace(/\s+/g, ' ')
            .replace(/^[\s,]+|[\s,]+$/g, '');
    }
    return args;
}
"""

# Replacing the prompt keeps its extra-network tags that the result does not already contain.
APPLY_JS = r"""
(text) => {
    const ta = gradioApp().querySelector('#TAB_prompt textarea');
    if (!ta || !text || !text.trim()) { return []; }
    const tags = [...new Set(ta.value.match(/<[^<>]+>/g) || [])].filter((tag) => !text.includes(tag));
    ta.value = [text.trim(), ...tags].join(' ');
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    return [];
}
"""

# txt2img and img2img build their own controls but share one settings store, so a
# change saved from one tab is never reverted by a later change in the other.
_SETTINGS = {"store": None}


def _settings_for(controls):
    if _SETTINGS["store"] is None:
        choices = {
            key: [choice[1] for choice in control.choices] + ([None] if control.value is None else [])
            for key, control in controls.items() if isinstance(control, gr.Dropdown)
        }
        ranges = {
            key: (control.minimum, control.maximum)
            for key, control in controls.items() if isinstance(control, gr.Slider)
        }
        _SETTINGS["store"] = SettingsStore(SETTINGS_FILE, {key: control.value for key, control in controls.items()},
                                           choices, ranges)
    return _SETTINGS["store"]


def _blocks(block):
    yield block
    for child in getattr(block, "children", None) or []:
        yield from _blocks(child)


def build_ui(tab):
    """Build the refiner inside an accordion of the txt2img or img2img tab."""
    preset_message = ""
    try:
        presets = load_presets()
    except (OSError, ValueError) as exc:
        presets = dict(DEFAULT_PRESETS)
        preset_message = f"Could not read presets.json: {exc}. The file was left unchanged; repair it or restore presets.json.bak before saving."
        print(f"[Prompt Refiner] {preset_message}", flush=True)
    preset_names = list(presets.keys())
    encoders = list_encoders()

    with gr.Accordion("Prompt Refiner", open=False, elem_id=f"prompt_refiner_{tab}_accordion") as accordion:
        with gr.Row():
            encoder = gr.Dropdown(
                label="Text encoder (models/text_encoder)", choices=encoders,
                value=encoders[0] if encoders else None, scale=10,
            )
            refresh_btn = gr.Button("🔄", elem_classes=["tool"], scale=1, min_width=40)
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Idea", lines=4,
                    placeholder="Leave empty to refine this tab's current prompt (<lora:...> tags are kept out and restored)",
                )
                with gr.Row():
                    run_btn = gr.Button("Refine prompt", variant="primary", scale=4)
                    stop_btn = gr.Button("Stop", scale=1)
            with gr.Column(scale=1):
                output = gr.Textbox(label="Result", lines=6, interactive=True, show_copy_button=True)
                with gr.Row():
                    apply_btn = gr.Button("Apply to prompt", variant="primary")
                    reuse_btn = gr.Button("Use result as idea")
        with gr.Row():
            preset = gr.Dropdown(label="System prompt preset", choices=preset_names,
                                 value=preset_names[0] if preset_names else None, scale=3)
            max_new = gr.Slider(32, 4096, value=768, step=16, label="Max new tokens", scale=2)
            empty_think = gr.Checkbox(value=True, label="Disable thinking (switchable Qwen3 / Qwen3.5 only)",
                                      info="Dedicated Thinking-2507 checkpoints cannot disable thinking.", scale=2)
        progress_html = gr.HTML(value="")
        status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Accordion("System prompt & presets", open=False):
            system_prompt = gr.Textbox(label="System prompt (instructions for the LLM)", lines=9,
                                       value=presets[preset_names[0]] if preset_names else "")
            with gr.Row():
                preset_name = gr.Textbox(label="Preset name", scale=3, placeholder="Name for a new preset",
                                         value=preset_names[0] if preset_names else "",
                                         info="Keep this name to update the preset, or change it to save a copy.")
                save_btn = gr.Button("Save preset", scale=1)
                delete_btn = gr.Button("Delete selected preset", scale=1)
            preset_status = gr.Textbox(label="Preset status", value=preset_message, interactive=False, lines=2)

        with gr.Accordion("Sampling options", open=False):
            gr.Markdown(
                "Defaults follow the **Qwen3-VL 4B Instruct VL profile**, including presence penalty **1.5** to discourage repetition. "
                "Use the natural-language system preset to preserve the idea's explicit details."
            )
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=SAMPLING_DEFAULTS["temperature"], step=0.05, label="Temperature (0 = greedy)")
                top_p = gr.Slider(0.0, 1.0, value=SAMPLING_DEFAULTS["top_p"], step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=SAMPLING_DEFAULTS["top_k"], step=1, label="Top-k (0 = off)")
                min_p = gr.Slider(0.0, 1.0, value=SAMPLING_DEFAULTS["min_p"], step=0.01, label="Min-p (0 = off)")
            with gr.Row():
                rep_penalty = gr.Slider(1.0, 2.0, value=SAMPLING_DEFAULTS["repetition_penalty"], step=0.01, label="Repetition penalty",
                                        info="Above 1 penalizes tokens in both the input and generated text.")
                presence = gr.Slider(0.0, 2.0, value=SAMPLING_DEFAULTS["presence_penalty"], step=0.05, label="Presence penalty (0 = off)",
                                     info="Discourages reusing generated tokens. The input idea and system instructions are excluded.")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            sampling_reset = gr.Button("Use Qwen3-VL defaults", size="sm")

        with gr.Accordion("Advanced model settings", open=False):
            with gr.Row():
                family = gr.Dropdown(models.FAMILIES, value="Auto", label="Model family",
                                     info="Override Auto for renamed encoders with ambiguous architecture.", scale=1)
                tokenizer_override = gr.Textbox(label="Tokenizer directory (optional)", scale=3,
                                                placeholder="Leave empty to use matching local Forge tokenizer files")
            with gr.Row():
                device = gr.Dropdown(["auto (GPU if available)", "cuda", "cpu"], value="auto (GPU if available)", label="Device")
                dtype = gr.Dropdown(["auto", "bfloat16", "float16", "float32"], value="auto", label="Compute dtype")
                rope = gr.Dropdown(["Auto (checkpoint configuration / model profile)", "1,000,000 (Qwen3)", "5,000,000 (Qwen3-VL)", "10,000,000 (Qwen3.5)"],
                                   value="Auto (checkpoint configuration / model profile)", label="RoPE theta")
            with gr.Row():
                keep_loaded = gr.Checkbox(value=False, label="Keep model loaded after generation (uses VRAM!)")
                free_vram = gr.Checkbox(value=True, label="Unload Forge's image models from VRAM before loading the LLM")
            with gr.Row():
                keep_q = gr.Checkbox(value=True, label="Keep supported quantized weights compressed (recommended)")
                compress = gr.Checkbox(value=False, label="Also compress bf16 / fp16 weights to int8 on load (reduces weight memory; may affect quality)")

        with gr.Accordion("LM head (for untied encoders with missing output weights)", open=False):
            gr.Markdown(
                "Known tied models need no extra weights. Qwen2.5-VL 7B, Qwen3.5 9B, and untied Qwen3/VL encoders need the exact "
                "`lm_head.weight` tensor from the original model on Hugging Face; it is stored once in "
                "`models/prompt_refiner/lm_heads/`. Select it explicitly unless its name matches the encoder stem or source metadata matches. "
                "Matching tensor dimensions alone do not establish compatibility."
            )
            with gr.Row():
                lm_choice = gr.Dropdown(label="LM head file", choices=["Auto"] + list_lm_heads(), value="Auto", scale=4)
                lm_refresh = gr.Button("🔄", elem_classes=["tool"], scale=1, min_width=40)
            gr.Markdown(
                "If refinement reports a missing output head: choose the standard original model below, "
                "click **Download lm_head**, then refine again with the downloaded file selected. "
                "For MOSS-Image / Qwen3.5 9B, the standard source is **Qwen/Qwen3.5-9B**. "
                "Enter a repository ID as `owner/model` (for example, `Qwen/Qwen3.5-9B`), "
                "not a web page URL or a diffusion model repository. For custom/fine-tuned encoders, "
                "enter the exact original checkpoint repository. Suggestions do not establish compatibility."
            )
            lm_source = gr.Dropdown(label="Known original model source", choices=list(KNOWN_HEAD_SOURCES),
                                    value=next(iter(KNOWN_HEAD_SOURCES)))
            with gr.Row():
                lm_repo = gr.Textbox(label="Hugging Face repo of the ORIGINAL model", placeholder="Qwen/Qwen3.5-9B", scale=4)
                lm_token = gr.Textbox(label="HF token (only for gated repos)", type="password", scale=2)
            with gr.Row():
                lm_int8 = gr.Checkbox(value=True, label="Store as int8 (half the size)")
                lm_btn = gr.Button("Download lm_head")
            lm_status = gr.HTML(value="")

        with gr.Accordion("Model details", open=False):
            model_details = gr.Textbox(label="Loaded model details", interactive=False, lines=5)
            unload_btn = gr.Button("Unload model")

        # Persist only explicitly listed controls. Prompt/result text, unsaved
        # instructions, preset-name drafts, download tokens, and status are excluded.
        controls = {
            "encoder": encoder, "preset": preset, "max_new_tokens": max_new,
            "disable_thinking": empty_think, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "min_p": min_p,
            "repetition_penalty": rep_penalty, "presence_penalty": presence, "seed": seed,
            "family": family, "tokenizer_directory": tokenizer_override,
            "device": device, "dtype": dtype, "rope": rope,
            "keep_loaded": keep_loaded, "free_vram": free_vram,
            "keep_quantized": keep_q, "compress": compress,
            "lm_head": lm_choice, "lm_head_int8": lm_int8, "lm_head_repo": lm_repo,
        }
        settings = _settings_for(controls)
        for key, control in controls.items():
            control.value = settings.values[key]
        system_prompt.value = presets.get(preset.value, "")
        preset_name.value = preset.value or ""
        lm_source.value = next((name for name, repo in KNOWN_HEAD_SOURCES.items() if repo == lm_repo.value), next(iter(KNOWN_HEAD_SOURCES)))
        settings_status = gr.Textbox(label="Settings status", value=settings.message,
                                     visible=bool(settings.message), interactive=False)

    def _remember(key):
        """Bind a single control to avoid overwriting other concurrent changes."""
        def changed(value):
            message = settings.update(key, value)
            return gr.update(value=message, visible=bool(message))
        return changed

    for key, control in controls.items():
        control.change(_remember(key), inputs=[control], outputs=[settings_status], show_progress="hidden")

    # ---- events ----
    sampling_reset.click(
        lambda: tuple(SAMPLING_DEFAULTS.values()),
        inputs=[], outputs=[controls[key] for key in SAMPLING_DEFAULTS],
        show_progress="hidden",
    )

    def _refresh(current):
        names = list_encoders()
        return gr.update(choices=names, value=current if current in names else (names[0] if names else None))

    refresh_btn.click(_refresh, inputs=[encoder], outputs=[encoder])
    lm_refresh.click(lambda cur: gr.update(choices=["Auto"] + list_lm_heads(), value=cur),
                     inputs=[lm_choice], outputs=[lm_choice])
    lm_source.input(lambda name: KNOWN_HEAD_SOURCES.get(name, ""), inputs=[lm_source], outputs=[lm_repo])
    lm_repo.input(lambda repo: next((name for name, value in KNOWN_HEAD_SOURCES.items() if value == repo), next(iter(KNOWN_HEAD_SOURCES))),
                  inputs=[lm_repo], outputs=[lm_source])
    lm_btn.click(create_lm_head_ui, inputs=[lm_repo, lm_int8, lm_token], outputs=[lm_status, lm_choice],
                 show_progress="hidden")

    # Only user selection reloads the editor; programmatic dropdown updates
    # must not clear the success/error message from a save or deletion.
    preset.input(select_preset_ui, inputs=[preset], outputs=[system_prompt, preset_name, preset_status])
    save_btn.click(change_preset_ui, inputs=[preset_name, system_prompt],
                   outputs=[preset, system_prompt, preset_name, preset_status])
    delete_btn.click(lambda name: change_preset_ui(name, delete=True), inputs=[preset],
                     outputs=[preset, system_prompt, preset_name, preset_status])

    def _run_with_details(*args):
        """Keep routine status short and put model diagnostics in their own panel."""
        for result, message, progress in run_refine(*args):
            summary, _, details = message.partition("\n")
            yield result, summary, progress, details

    run_btn.click(
        _run_with_details,
        inputs=[encoder, prompt, system_prompt, temperature, top_p, top_k, min_p, rep_penalty,
                max_new, seed, device, dtype, rope, keep_loaded, free_vram, empty_think, keep_q, compress, lm_choice, family, tokenizer_override, presence],
        outputs=[output, status, progress_html, model_details],
        show_progress="hidden",
        js=FILL_IDEA_JS.replace("TAB", tab),
    )
    stop_btn.click(lambda: _STOP.set(), inputs=None, outputs=None)
    reuse_btn.click(lambda t: t, inputs=[output], outputs=[prompt])
    unload_btn.click(lambda: (unload_model(), ""), outputs=[status, model_details])
    apply_btn.click(fn=None, inputs=[output], outputs=None, js=APPLY_JS.replace("TAB", tab))

    # Forge applies ui-config.json after the tab is built. Its older defaults must
    # not overwrite settings.json or restore text/secrets that this extension
    # deliberately excludes from persistence, so every control opts out.
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
