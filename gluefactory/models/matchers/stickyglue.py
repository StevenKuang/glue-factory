import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import Rearrange, reduce, repeat
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
import matplotlib.pyplot as plt

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics

from .sticky_mem import SimpleTransformer, StickyMemory

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )

        self.loss_fn = NLLLoss(conf.loss)

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                fname = (
                    f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                    + ".pth"
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    self.url.format(conf.weights_from_version, conf.weights),
                    file_name=fname,
                )

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

            # only for eval
            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        # pasted from lightglue standalone code for finding mutual matches
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "matches": matches,
            "scores": mscores,
            "prune0": prune0,
            "prune1": prune1,
        }

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights

        # confidences
        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics

    
class SpatialMemory():
    def __init__(self, max_num_matches, max_frames, dim, norm_q, norm_k, norm_v, device, mem_dropout=None, 
                 long_mem_size=4000, work_mem_size=5, 
                 attn_thresh=5e-4, sim_thresh=0.95, 
                 save_attn=False, num_patches=None):
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_v = norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.top_k = long_mem_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.num_patches = num_patches
        self.max_num_matches = max_num_matches
        self.max_frames = max_frames
        self.pair_proj = None
        self.device = device

        self.merge_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        ).to(self.device)
        self.init_mem()
    
    def init_mem(self):
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.mem_imgs = None
        self.lm = 0
        self.wm = 0
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat):
        if self.mem_k is None:
            self.mem_k = feat
        else:
            self.mem_k = torch.cat((self.mem_k, feat), dim=1)

        return self.mem_k
    
    def add_mem_v(self, feat):
        if self.mem_v is None:
            self.mem_v = feat
        else:
            self.mem_v = torch.cat((self.mem_v, feat), dim=1)

        return self.mem_v

    def add_mem_c(self, feat):
        if self.mem_c is None:
            self.mem_c = feat
        else:
            self.mem_c = torch.cat((self.mem_c, feat), dim=1)

        return self.mem_c
    
    def add_mem_pts(self, pts_cur):
        if pts_cur is not None:
            if self.mem_pts is None:
                self.mem_pts = pts_cur
            else:
                self.mem_pts = torch.cat((self.mem_pts, pts_cur), dim=1)
    
    def add_mem_img(self, img_cur):
        if img_cur is not None:
            if self.mem_imgs is None:
                self.mem_imgs = img_cur
            else:
                self.mem_imgs = torch.cat((self.mem_imgs, img_cur), dim=1)

    
    def check_sim(self, feat_k, thresh=0.7):
        # Do correlation with working memory
        if self.mem_k is None or thresh==1.0:
            return False
        
        wmem_size = self.wm * self.num_patches

        # wm: BS, T, 196, C
        wm = self.mem_k[:, -wmem_size:].reshape(self.mem_k.shape[0], -1, self.num_patches, self.mem_k.shape[-1])

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        wm_norm = F.normalize(wm, p=2, dim=-1)

        corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)

        mean_corr = torch.mean(corr, dim=-1)

        if mean_corr.max() > thresh:
            print('Similarity detected:', mean_corr.max())
            return True
    
        return False

    def merge_layer(self, features):
        """
        Merge old and new processed features
        features: [2, d] (old and new)
        """
        return self.merge_mlp(features.mean(dim=0))
    
    def add_memory(self, data, matches, scores):
        # matches: list of tensors [batch][k, 2] with variable k
        # scores: list of tensors [batch][k] with variable k
        desc0 = data['descriptors0']  # [bs, n, d]
        desc1 = data['descriptors1']  # [bs, m, d]
        bs = desc0.shape[0]

        # Pad matches to max_num_matches with -1
        padded_matches = [
            F.pad(m, (0, 0, 0, self.max_num_matches - m.shape[0]), value=-1)
            for m in matches
        ]
        padded_matches = torch.stack(padded_matches)  # [bs, max_k, 2]
        
        # Create mask for valid matches
        valid_mask = (padded_matches[..., 0] != -1) & (padded_matches[..., 1] != -1)
        
        # Gather descriptor pairs with padding
        desc_pairs = torch.stack([
            torch.where(valid_mask.unsqueeze(-1), 
                      desc0[torch.arange(bs)[:, None], padded_matches[..., 0].clamp(min=0)], 
                      torch.zeros_like(desc0[:, :1])),
            torch.where(valid_mask.unsqueeze(-1),
                      desc1[torch.arange(bs)[:, None], padded_matches[..., 1].clamp(min=0)],
                      torch.zeros_like(desc1[:, :1]))
        ], dim=2)  # [bs, max_k, 2, d]

        # Process all pairs at once
        processed_desc = self.process_matches(desc_pairs)  # [bs, max_k, d]
        
        # Apply valid mask to zero out padded processing results
        processed_desc = processed_desc * valid_mask.unsqueeze(-1).to(processed_desc.dtype)
        
        
        # TODO: Update memory, update attention and count
        # Initialize memory if empty
        if self.mem_k is None:
            self.mem_k = torch.zeros(bs, self.max_num_matches, desc0.shape[-1], device=desc0.device)
            self.mem_v = torch.zeros(bs, self.max_num_matches, desc0.shape[-1], device=desc0.device)
            self.desc_count = torch.zeros(bs, self.max_num_matches, device=desc0.device)  # Track number of descriptors per point
            
        # For each batch
        for b in range(bs):
            # For each valid match in current batch
            curr_valid = valid_mask[b]
            for i in range(curr_valid.sum()):
                match = padded_matches[b, i]  # Current match pair
                desc_pair = desc_pairs[b, i]  # Current descriptor pair
                proc_desc = processed_desc[b, i]  # Processed descriptor
                
                # Find if this point was matched before
                # We can use descriptor similarity to find corresponding points
                similarity = torch.einsum('nd,kmd->km', 
                    desc_pair,  # [2, d]
                    self.mem_k[b, :, :self.max_descs_per_point])  # [num_kpts, num_descs, d]
                

                # Get max similarity across all stored descriptors for each point
                max_sim, _ = similarity.max(dim=1)  # [num_kpts]
                
                if max_sim.max() > self.sim_thresh:
                    # Point exists in memory, update it
                    point_idx = max_sim.argmax()
                    
                    # Update key (add new descriptors)
                    curr_count = int(self.desc_count[b, point_idx])
                    if curr_count < self.max_descs_per_point:
                        self.mem_k[b, point_idx, curr_count:curr_count+2] = desc_pair
                        self.desc_count[b, point_idx] += 2
                    
                    # Update value (merge with existing)
                    self.mem_v[b, point_idx] = self.merge_layer(
                        torch.stack([self.mem_v[b, point_idx], proc_desc])
                    )
                else:
                    # New point, find empty slot
                    empty_idx = (self.desc_count[b] == 0).nonzero(as_tuple=True)[0]
                    if len(empty_idx) > 0:
                        idx = empty_idx[0]
                        # Add new key
                        self.mem_k[b, idx, 0:2] = desc_pair
                        self.desc_count[b, idx] = 2
                        # Add new value
                        self.mem_v[b, idx] = proc_desc
    
        # Prune if needed
        if self.mem_k.size(1) > self.long_mem_size:
            self.prune_memory()

    def process_matches(self, desc_pairs):
        """
        desc_pairs: [b, k, 2, d]
        """
        b, k, _, d = desc_pairs.shape
        desc_concat = desc_pairs.view(b, k, 2*d)
        if self.pair_proj is None:
            self.pair_proj = nn.Linear(2*d, d).to(desc_pairs.device)
        return self.pair_proj(desc_concat)  # [b, k, d]
    
    def memory_read(self, data, res=False):
        desc0 = data['descriptors0']
        desc1 = data['descriptors1']
        data['descriptors0'] = self.memory_read_per_desc(desc0, res)
        data['descriptors1'] = self.memory_read_per_desc(desc1, res)
        return data

    def memory_read_per_desc(self, query_feat, res=False):
        """Memory read with query attention mechanism
        Args:
            query_feat: [bs, num_kpts, d] - current frame descriptors
            res: whether to add residual connection
        Returns:
            enhanced features: [bs, num_kpts, d]
        """
        if self.mem_k is None:
            return query_feat

        # query_feat: [bs, num_kpts, d]
        # mem_k: [bs, max_num_matches*max_frames, d], N = max_num_matches*max_frames
        # mem_v: [bs, max_num_matches*max_frames, d]

        # Compute attention between query and all stored descriptors
        #TODO: fix the situation where the bs is not full
        affinity = torch.einsum('bnd,bmd->bnm',
            self.norm_q(query_feat),  # [bs, num_kpts, d]
            self.norm_k(self.mem_k)       # [bs, N, d]
        )

        affinity = affinity / math.sqrt(query_feat.shape[-1])
        
        # Compute attention weights
        attn = torch.softmax(affinity, dim=-1)  # [bs, num_kpts, N]
        
        # Apply attention to memory values
        enhanced_feat = torch.einsum('bnm,bmd->bnd',
            attn,                  # [bs, num_kpts, N]
            self.norm_v(self.mem_v)  # [bs, N, d]
        )
        
        # # Update attention statistics
        # if self.mem_attn is None:
        #     self.mem_attn = torch.zeros(attn.shape, device=attn.device)
        # self.mem_attn += torch.sum(attn, dim=1, keepdim=True)  
        
        if res:
            enhanced_feat = enhanced_feat + query_feat  # [bs, num_kpts, d]
            
        return enhanced_feat
    
    def memory_prune(self):

        weights = self.mem_attn / self.mem_count
        weights[self.mem_count<self.work_mem_size+5] = 1e8

        num_mem_b = self.mem_k.shape[1]


        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=1)
        top_k_indices_expanded = top_k_indices.expand(-1, -1, self.mem_k.size(-1))


        self.mem_k = torch.gather(self.mem_k, -2, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, -2, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, -2, top_k_indices)
        self.mem_count = torch.gather(self.mem_count, -2, top_k_indices)
 

        if self.mem_pts is not None:
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 256, 3)
            self.mem_pts = torch.gather(self.mem_pts, 1, top_k_indices_expanded)
            self.mem_imgs = torch.gather(self.mem_imgs, 1, top_k_indices_expanded)

        num_mem_a = self.mem_k.shape[1]

        print('Memory pruned:', num_mem_b, '->', num_mem_a)
  

class StickyGlue(nn.Module):
    default_conf = {
        **LightGlue.default_conf,
        "name": "stickyglue",
        "neural_memory": {
            "enabled": True,
            "depth": 2,  # NEURAL_MEMORY_DEPTH
            "dim_head": 64,
            "heads": 4,
            "attn_pool_chunks": True,  # STORE_ATTN_POOL_CHUNKS
            "learned_mem_model_weights": True,
            "aux_kv_recon_loss_weight": 0.1,  # Weight for memory reconstruction loss
            "mem_dropout": None,
            "max_frames": 5
        },
        "pretrained_lightglue": "",  # Path to pretrained LightGlue checkpoint
        "is_training": True
    }

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        
        # Initialize LightGlue as the base matcher
        if conf.pretrained_lightglue:
            conf.weights = conf.pretrained_lightglue
        self.lightglue = LightGlue(conf)
        
        self.batch_size = conf.memory_training.batch_size
        self.max_num_matches = conf.memory_training.max_num_matches
        self.is_training = conf.is_training
        self.max_frame = conf.max_frame

        self.norm_q = nn.LayerNorm(conf.neural_memory.dim_head * conf.neural_memory.heads)
        self.norm_k = nn.LayerNorm(conf.neural_memory.dim_head * conf.neural_memory.heads)
        self.norm_v = nn.LayerNorm(conf.neural_memory.dim_head * conf.neural_memory.heads)
        self.mem_dropout = conf.neural_memory.mem_dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.is_training:
            self.spa_mem = SpatialMemory(self.max_num_matches,
                                self.max_frames,
                                conf.neural_memory.dim_head * conf.neural_memory.heads,
                                self.norm_q, self.norm_k, self.norm_v, 
                                device=self.device,
                                mem_dropout=self.mem_dropout, 
                                attn_thresh=0)
        else:
            self.spa_mem = SpatialMemory(self.max_num_matches,
                                conf.neural_memory.dim_head * conf.neural_memory.heads,
                                self.norm_q, self.norm_k, self.norm_v, 
                                device=self.device)

    # def forward(self, data: dict) -> dict:       # for inference

    #     # # save view0 and view1
    #     # import matplotlib.pyplot as plt
    #     # for b in range(data['descriptors0'].shape[0]):
    #     #     plt.imshow(data['view0']['image'][b].detach().cpu().permute(1, 2, 0).numpy())
    #     #     plt.savefig(f"view0.png")
    #     #     plt.imshow(data['view1']['image'][b].detach().cpu().permute(1, 2, 0).numpy())
    #     #     plt.savefig(f"view1.png")

    #     if self.is_training:
    #         return self.train_forward(data)
        
    #     b, n ,d = data['descriptors0'].shape
    #     device = data['descriptors0'].device
    #     _data = data.copy()
    #     if b < self.batch_size:
    #         # pad the batch to the batch size, the first dim is the batch size
    #         _data['descriptors0'] = F.pad(data['descriptors0'], (0, 0, 0, 0, 0, self.batch_size - b))
    #         _data['descriptors1'] = F.pad(data['descriptors1'], (0, 0, 0, 0, 0, self.batch_size - b))


    #     if self.gru is None:
    #         self.gru = nn.GRU(input_size=d, hidden_size=d, num_layers=1, batch_first=True).to(device)
    #         self.h = torch.zeros(1, b, d, device=device)
    #         self.ff = nn.Linear(d, d).to(device)
    #     else:
    #         with torch.no_grad():
    #             _data['descriptors0']= self.ff(self.gru(_data['descriptors0'], self.h)[0])
    #             _data['descriptors1']= self.ff(self.gru(_data['descriptors1'], self.h)[0])

    #     if b < self.batch_size:
    #         # keep the first b elements
    #         _data['descriptors0'] = _data['descriptors0'][:b,:,:]
    #         _data['descriptors1'] = _data['descriptors1'][:b,:,:]

    #     pred = self.lightglue(_data)

    #     # update gru hidden state

    #     if b < self.batch_size:
    #         # pad again
    #         _data['descriptors0'] = F.pad(_data['descriptors0'], (0, 0, 0, 0, 0, self.batch_size - b))
    #         _data['descriptors1'] = F.pad(_data['descriptors1'], (0, 0, 0, 0, 0, self.batch_size - b))

    #     gru_desc0, self.h = self.gru(_data['descriptors0'], self.h)
    #     gru_desc1, self.h = self.gru(_data['descriptors1'], self.h)
    #     gru_desc0 = self.ff(gru_desc0)
    #     gru_desc1 = self.ff(gru_desc1)


    #     if b < self.batch_size:
    #         gru_desc0 = gru_desc0[:b,:,:]
    #         gru_desc1 = gru_desc1[:b,:,:]

    #     gru_loss = self.gru_loss(gru_desc0, pred['ref_descriptors0'][:,-1]) + self.gru_loss(gru_desc1, pred['ref_descriptors1'][:,-1])
    #     pred['loss/memory'] = gru_loss.expand(b)

    #     return pred


    def forward(self, data: dict) -> dict:
        # # save view0 and view1
        # for b in range(data['descriptors0'].shape[0]):
        #     plt.imshow(data['view0']['image'][b].detach().cpu().permute(1, 2, 0).numpy())
        #     plt.savefig(f"view0.png")
        #     plt.imshow(data['view1']['image'][b].detach().cpu().permute(1, 2, 0).numpy())
        #     plt.savefig(f"view1.png")

        b, n ,d = data['descriptors0'].shape
        device = data['descriptors0'].device
        _data = data.copy()

        _data = self.spa_mem.memory_read(_data, res=True)   

        pred = self.lightglue(_data)

        matches, scores = pred['matches'], pred['scores']

        # mem_loss = self.spa_mem.add_memory(data, matches, scores)
        self.spa_mem.add_memory(data, matches, scores)

        # pred['loss/memory'] = mem_loss

        return pred

    def loss(self, pred, data):
        losses, metrics = self.lightglue.loss(pred, data)
        if "loss/memory" in pred:
            losses["memory"] = pred["loss/memory"]
            if self.training:
                losses["total"] += losses["memory"]
        return losses, metrics

__main_model__ = StickyGlue