"""Keep input embeddings in system RAM while the rest of the model runs on the GPU.

An untied model's input embedding table is only indexed: each token reads one row.
Keeping it in RAM frees its full size in VRAM (about 2 GB for Qwen3.5 9B in BF16)
at the cost of moving the token ids to the CPU and the looked-up rows back, a few
kilobytes per generated token. Tied models share this table with the output head,
which multiplies by all of it for every token, so they keep it on the GPU.
"""

import torch


class HostEmbedding(torch.nn.Module):
    """An embedding whose table stays on the CPU through model.to(...) calls."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().to("cpu"), requires_grad=False)
        self.num_embeddings, self.embedding_dim = self.weight.shape

    def _apply(self, fn, *args, **kwargs):
        # model.to(device) reaches every submodule through _apply; this table stays put.
        return self

    def forward(self, input_ids):
        rows = torch.nn.functional.embedding(input_ids.to("cpu"), self.weight)
        return rows.to(input_ids.device, non_blocking=True)


def host_embeddings(model):
    """Move an untied model's input embeddings to RAM; return the bytes freed from the device."""
    embed = model.model.embed_tokens
    if isinstance(embed, HostEmbedding):
        return 0
    lm_head = getattr(getattr(model, "lm_head", None), "weight", None)
    if lm_head is not None and lm_head.untyped_storage().data_ptr() == embed.weight.untyped_storage().data_ptr():
        return 0
    model.model.embed_tokens = HostEmbedding(embed.weight)
    return embed.weight.numel() * embed.weight.element_size()
