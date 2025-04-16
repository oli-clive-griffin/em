# some stuff for comparing activations and KL divergences
from collections.abc import Iterable
import gc
from typing import cast
import torch
import numpy as np
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens.HookedTransformer import TransformerBlock  # type: ignore
from dataclasses import dataclass


def normalised_distance(
    acts_base_SLD: np.ndarray, acts_tuned_SLD: np.ndarray
) -> np.ndarray:
    tuned_norms_SL = np.linalg.norm(acts_tuned_SLD, ord=2, axis=-1)
    base_norms_SL = np.linalg.norm(acts_base_SLD, ord=2, axis=-1)
    mean_l2_norm_SL = (tuned_norms_SL + base_norms_SL) / 2
    nmse_SL = np.linalg.norm(acts_tuned_SLD - acts_base_SLD, axis=-1) / mean_l2_norm_SL
    return nmse_SL


def _get_logits_and_resid(prompt: str, model: HookedTransformer, hookpoints: list[str]):
    toks_1S: torch.Tensor = model.tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
    assert toks_1S.shape[0] == 1
    seq_logits, cache = model.run_with_cache(
        toks_1S,
        names_filter=lambda name: name in hookpoints,
    )
    toks_S = toks_1S[0]
    seq_logits_SV = seq_logits[0]
    cache_ = cache.remove_batch_dim()
    resid_SLD = torch.stack([cache_.cache_dict[hp] for hp in hookpoints]).transpose(
        0, 1
    )
    return seq_logits_SV, resid_SLD, toks_S


def tokenwise_kl(probs_P_SV: torch.Tensor, probs_Q_SV: torch.Tensor):
    """S = seq, V = vocab"""
    tokens_kl_S = torch.sum(probs_P_SV * torch.log(probs_P_SV / probs_Q_SV), dim=-1)
    return tokens_kl_S


@dataclass
class SeqData:
    input_tokens_S: torch.Tensor
    base_pred_toks_S: torch.Tensor
    tuned_pred_toks_S: torch.Tensor
    kl_div_S: torch.Tensor
    acts_base_SLD: torch.Tensor
    acts_tuned_SLD: torch.Tensor


def run_acts_through_other_model(
    resid_mid_SLD: torch.Tensor, other_model: HookedTransformer
) -> torch.Tensor:
    resid_post_SLD = torch.zeros_like(resid_mid_SLD)
    seq_len, n_layers, dim = resid_mid_SLD.shape
    for l, block in enumerate(cast(Iterable[TransformerBlock], other_model.blocks)):
        # block.hook_mlp_out
        mlp_out_1SD = block.apply_mlp(block.ln2(resid_mid_SLD[:, l, :][None]))
        assert mlp_out_1SD.shape == (1, seq_len, dim)
        resid_post_SLD[:, l, :] = mlp_out_1SD[0]
    return resid_post_SLD


def get_seq_data(
    prompt: str,
    llm_base: HookedTransformer,
    llm_tuned: HookedTransformer,
    hookpoints: list[str],
) -> SeqData:
    base_logits_SV, base_resid_SLD, base_toks_S = _get_logits_and_resid(
        prompt, llm_base, hookpoints
    )
    tuned_logits_SV, tuned_resid_SLD, tuned_toks_S = _get_logits_and_resid(
        prompt, llm_tuned, hookpoints
    )

    assert (base_toks_S == tuned_toks_S).all()
    input_tokens_S = base_toks_S

    base_seq_probs_SV = base_logits_SV.softmax(dim=-1)
    tuned_seq_probs_SV = tuned_logits_SV.softmax(dim=-1)

    base_seq_preds_S = base_logits_SV.argmax(dim=-1)
    tuned_seq_preds_S = tuned_logits_SV.argmax(dim=-1)

    kl_div_S = tokenwise_kl(probs_P_SV=base_seq_probs_SV, probs_Q_SV=tuned_seq_probs_SV)

    return SeqData(
        input_tokens_S=input_tokens_S,
        base_pred_toks_S=base_seq_preds_S,
        tuned_pred_toks_S=tuned_seq_preds_S,
        kl_div_S=kl_div_S,
        acts_base_SLD=base_resid_SLD,
        acts_tuned_SLD=tuned_resid_SLD,
    )


def clear_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
