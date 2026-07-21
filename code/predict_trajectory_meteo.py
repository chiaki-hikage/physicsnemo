"""
Flow Matching モデルによる軌跡予測

推論パイプライン:
1. VLM（Cosmos-Reason2-2B, Stage 1 学習済み）で画像+履歴軌跡を処理
2. VLM generate() で CoT を自己回帰生成 → <|traj_future_start|> で停止 → KV cache 取得
3. Flow Matching（Euler integration）で連続アクション（64, 2）をサンプリング
4. UnicycleAccelCurvatureActionSpace.action_to_traj() で軌跡に変換
"""

import sys
import os
import math
import copy
import logging
from typing import Any

import torch
import torch.nn as nn
import einops
from PIL import Image  # !!! 追加: 天候分類の画像前処理用
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Qwen3VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoImageProcessor,  # !!! 追加: 天候分類モデル用
    SiglipForImageClassification,  # !!! 追加: 天候分類モデル用
)
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
)

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "alpamayo1.5",
        "src",
    )
)

from alpamayo1_5.action_space.unicycle_accel_curvature import (
    UnicycleAccelCurvatureActionSpace,
)
from alpamayo1_5.models.delta_tokenizer import DeltaTrajectoryTokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# 定数
# =============================================================================

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

SPECIAL_TOKENS = {
    k: "<|" + k + "|>"
    for k in [
        "prompt_start",
        "prompt_end",
        "image_start",
        "image_pre_tkn",
        "image_end",
        "traj_history_start",
        "traj_history_pre_tkn",
        "traj_history_end",
        "cot_start",
        "cot_end",
        "meta_action_start",
        "meta_action_end",
        "traj_future_start",
        "traj_future_pre_tkn",
        "traj_future_end",
        "traj_history",
        "traj_future",
        "image_pad",
        "vectorized_wm",
        "vectorized_wm_start",
        "vectorized_wm_end",
        "vectorized_wm_pre_tkn",
        "route_start",
        "route_pad",
        "route_end",
        "question_start",
        "question_end",
        "answer_start",
        "answer_end",
    ]
}


# =============================================================================
# !!! 追加: 天候ガード（豪雨検知 → CoT強制挿入 / 速度クランプ）関連の定数
# =============================================================================

# prithivMLmods/Weather-Image-Classification の id2label を
# weather_classify/weather_classifier_eval2.py の map_weather_to_binary() と
# 同じ規則で clear / heavy_rain / unknown に丸める
WEATHER_MODEL_NAME = "prithivMLmods/Weather-Image-Classification"

# 豪雨検出時にCoTへ強制的に挿入する固定文
# （自由生成には頼らず、決定的にこの文を差し込む）
HEAVY_RAIN_COT_MESSAGE = (
    "Heavy rain is detected and the road is wet with reduced visibility, "
    "so the vehicle reduces its speed and drives cautiously for safety."
)


# =============================================================================
# 推論用 Config
# =============================================================================

class _InferenceConfig:
    """推論用設定（210/800 と同一パラメータ）"""

    def __init__(self, model_variant: str = "base"):
        self.VLM_MODEL = "nvidia/Cosmos-Reason2-2B"

        self.STAGE1_CHECKPOINT_DIR = (
            "/home/username/alpamayo_train/outputs/"
            "vlm/action_modality_injection_cot/300000_w_nvidia_cot"
        )  # VLM

        if model_variant == "base":
            self.STAGE1_CHECKPOINT_DIR = self.STAGE1_CHECKPOINT_DIR
            self.SAVE_DIR = (
                "/home/username/alpamayo_train/outputs/flow_matching/coc/300000/checkpoint"
            )  # Flow Matching
        else:
            self.STAGE1_CHECKPOINT_DIR = (
                f"{self.STAGE1_CHECKPOINT_DIR}_{model_variant}"
            )
            self.SAVE_DIR = (
                "/home/username/alpamayo_train/outputs/flow_matching/coc/300000_" + model_variant
            )  # Flow Matching

        # Expert Model
        self.EXPERT_HIDDEN_SIZE = 1024
        self.EXPERT_INTERMEDIATE_SIZE = 4128
        self.EXPERT_NUM_ATTENTION_HEADS = 8
        self.EXPERT_HEAD_DIM = 128
        self.EXPERT_NON_CAUSAL_ATTENTION = True

        # Action In Proj
        self.ACTION_IN_PROJ_HIDDEN_SIZE = 512
        self.ACTION_IN_PROJ_MAX_FREQ = 100.0
        self.ACTION_IN_PROJ_NUM_ENC_LAYERS = 2
        self.ACTION_IN_PROJ_NUM_FOURIER_FEATS = 20

        # Flow Matching
        self.NUM_INFERENCE_STEPS = 10

        # Action Space
        self.NUM_FUTURE_WAYPOINTS = 64
        self.DT = 0.1
        self.ACCEL_MEAN = 0.02902694707164455
        self.ACCEL_STD = 0.6810426736454882
        self.CURVATURE_MEAN = 0.0002692167976330542
        self.CURVATURE_STD = 0.026148280660833106
        self.ACCEL_BOUNDS = (-9.8, 9.8)
        self.CURVATURE_BOUNDS = (-0.33, 0.33)

        # Token
        self.FUTURE_NUM_BINS = 3000
        self.HIST_NUM_BINS = 1000
        self.TRAJ_VOCAB_SIZE = self.FUTURE_NUM_BINS + self.HIST_NUM_BINS
        self.TOKENS_PER_HISTORY_TRAJ = 48  # 16 steps * 3 (Δx, Δy, Δz)

        # Image
        self.MIN_PIXELS = 163840
        self.MAX_PIXELS = 196608

        # Generate（CoT 生成）
        self.MAX_COT_GENERATION_LENGTH = 256
        self.GENERATE_TOP_P = 0.99
        self.GENERATE_TEMPERATURE = 0.1
        self.GENERATE_DO_SAMPLE = True
        # self.GENERATE_DO_SAMPLE = False

        # !!! 追加: 天候ガード（豪雨検知 → CoT強制挿入 / 速度クランプ）
        self.WEATHER_MODEL_NAME = WEATHER_MODEL_NAME
        # 天候分類器の予測confidenceがこの値未満の場合は豪雨判定にしない
        self.WEATHER_CONFIDENCE_THRESHOLD = 0.6
        # 豪雨時の速度上限（ユーザー指定: 10km/h）
        self.V_MAX_RAIN_KMH = 10.0
        self.V_MAX_RAIN_MPS = self.V_MAX_RAIN_KMH / 3.6
        self.HEAVY_RAIN_COT_MESSAGE = HEAVY_RAIN_COT_MESSAGE

        self.DTYPE = torch.bfloat16
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# モデルコンポーネント定義（210/800 と同一）
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FourierEncoderV2(nn.Module):
    def __init__(self, dim: int, max_freq: float = 100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(
            0,
            math.log10(max_freq),
            steps=half,
        )
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat(
            [torch.sin(arg), torch.cos(arg)],
            -1,
        ) * math.sqrt(2)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        num_input_feats: int,
        num_enc_layers: int,
        hidden_size: int,
        outdim: int,
    ):
        super().__init__()
        assert num_enc_layers >= 1

        layers = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]

        for i in range(num_enc_layers):
            if i < num_enc_layers - 1:
                layers.extend(
                    [
                        RMSNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                layers.extend(
                    [
                        RMSNorm(hidden_size),
                        nn.Linear(hidden_size, outdim),
                    ]
                )

        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class PerWaypointActionInProjV2(nn.Module):
    def __init__(
        self,
        in_dims: list[int] | tuple[int, ...],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        super().__init__()

        self.in_dims = list(in_dims)
        self.out_dim = out_dim

        self.sinus = nn.ModuleList(
            [
                FourierEncoderV2(
                    dim=num_fourier_feats,
                    max_freq=max_freq,
                )
                for _ in range(in_dims[-1])
            ]
        )

        self.timestep_fourier_encoder = FourierEncoderV2(
            dim=num_fourier_feats,
            max_freq=max_freq,
        )

        num_input_feats = (
            sum(s.out_dim for s in self.sinus)
            + self.timestep_fourier_encoder.out_dim
        )

        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        action_feats = torch.cat(
            [
                s(x[:, :, i])
                for i, s in enumerate(self.sinus)
            ],
            dim=-1,
        )

        timestep_feats = self.timestep_fourier_encoder(
            timesteps[..., -1]
        )
        timestep_feats = timestep_feats.repeat(1, T, 1)

        combined = torch.cat(
            (
                action_feats,
                timestep_feats,
            ),
            dim=-1,
        )

        encoded = self.encoder(combined.flatten(0, 1))
        return self.norm(encoded.reshape(B, T, -1))


# =============================================================================
# ユーティリティ関数（210 準拠）
# =============================================================================

def _replace_pad_token(
    input_ids: torch.Tensor,
    new_ids: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    mask = input_ids == pad_idx
    return input_ids.masked_scatter(mask, new_ids)


class _ExpertLogitsProcessor(LogitsProcessor):
    """離散軌跡トークンの logits をマスクし、CoT 生成時にテキストのみ生成させる"""

    def __init__(
        self,
        traj_token_offset: int,
        traj_vocab_size: int,
    ):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        scores[
            :,
            self.traj_token_offset : self.traj_token_offset
            + self.traj_vocab_size,
        ] = float("-inf")
        return scores


class _StopAfterEOS(StoppingCriteria):
    """<|traj_future_start|> 出力後に +1 token で停止（KV cache 更新のため）"""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.eos_found = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        batch_size = input_ids.shape[0]

        if self.eos_found is None:
            self.eos_found = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=input_ids.device,
            )

        if self.eos_found.all():
            return True

        last_tokens = input_ids[:, -1]
        self.eos_found = self.eos_found | (
            last_tokens == self.eos_token_id
        )

        return False


def _replace_padding_after_eos(
    token_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """最初の EOS トークン以降を padding で上書きする"""

    batch_size, seq_len = token_ids.shape
    eos_mask = token_ids == eos_token_id

    eos_positions = torch.where(
        eos_mask,
        torch.arange(
            seq_len,
            device=token_ids.device,
        ).unsqueeze(0).expand(batch_size, -1),
        torch.tensor(
            seq_len,
            device=token_ids.device,
        ),
    )

    first_eos_pos = eos_positions.min(
        dim=1,
        keepdim=True,
    )[0]

    position_indices = torch.arange(
        seq_len,
        device=token_ids.device,
    ).unsqueeze(0)

    mask_after = position_indices > first_eos_pos
    token_ids = token_ids.clone()
    token_ids[mask_after] = pad_token_id

    return token_ids


# =============================================================================
# !!! 追加: 天候分類（豪雨検知）
#
# weather_classify/weather_classifier_eval2.py で評価済みの
# prithivMLmods/Weather-Image-Classification（SiglipForImageClassification）
# をそのまま利用する。VLM/Expertとは完全に独立したfrozenモデルで、
# KV cache・勾配には一切関与しない。
# =============================================================================

def _map_weather_label_to_binary(label: str) -> str:
    """weather_classify/weather_classifier_eval2.py の map_weather_to_binary() 準拠。

    モデルのid2labelを clear / heavy_rain / unknown に丸める。
    """
    s = label.lower()

    if "rain" in s:
        return "heavy_rain"

    if "clear" in s or "sun" in s or "shine" in s:
        return "clear"

    # cloudy/overcast は clear ではないが、豪雨でもない。
    # 2値評価ではいったん clear 側に倒す。
    if "cloud" in s or "overcast" in s:
        return "clear"

    # foggy/hazy は豪雨ではないが悪条件のため、低速走行側に倒す。
    if "fog" in s or "hazy" in s:
        return "heavy_rain"

    return "unknown"


@torch.no_grad()
def _classify_weather(
    cfg: "_InferenceConfig",
    frame_uint8: torch.Tensor,
) -> tuple[bool, str, float]:
    """カメラ画像1枚（豪雨判定用）から天候を分類する。

    Args:
        frame_uint8: shape=(3, H, W), dtype=torch.uint8

    Returns:
        is_heavy_rain: confidence がしきい値以上の豪雨判定かどうか
        pred_label: 分類器が出力した生ラベル
        confidence: 予測確信度
    """
    image = Image.fromarray(
        frame_uint8.permute(1, 2, 0).cpu().numpy()
    )

    inputs = _state.weather_processor(
        images=[image],
        return_tensors="pt",
    ).to(cfg.DEVICE)

    outputs = _state.weather_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_idx = int(probs.argmax().item())
    pred_label = _state.weather_model.config.id2label[pred_idx]
    confidence = float(probs[pred_idx].item())

    is_heavy_rain = (
        _map_weather_label_to_binary(pred_label) == "heavy_rain"
        and confidence >= cfg.WEATHER_CONFIDENCE_THRESHOLD
    )

    return is_heavy_rain, pred_label, confidence


# =============================================================================
# VLM KV キャッシュ取得（generate ベース、210/alpamayo1_5.py 準拠）
# =============================================================================
# <|cot_start|> から CoT を自己回帰生成し
# <|traj_future_start|> 出力後に停止 → KV cache を返す
# =============================================================================

@torch.no_grad()
def _get_vlm_kv_cache(
    cfg: _InferenceConfig,
    vlm: nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    traj_future_start_id: int,
    traj_token_start_idx: int,
    traj_vocab_size: int,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    extra_kwargs = {}

    if pixel_values is not None:
        extra_kwargs["pixel_values"] = pixel_values.to(dtype=dtype)

    if image_grid_thw is not None:
        extra_kwargs["image_grid_thw"] = image_grid_thw

    generation_config = vlm.generation_config
    generation_config.top_p = cfg.GENERATE_TOP_P
    generation_config.temperature = cfg.GENERATE_TEMPERATURE
    generation_config.do_sample = cfg.GENERATE_DO_SAMPLE
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = cfg.MAX_COT_GENERATION_LENGTH
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = tokenizer.pad_token_id

    stopping_criteria = StoppingCriteriaList(
        [
            _StopAfterEOS(
                eos_token_id=traj_future_start_id,
            )
        ]
    )

    logits_processor = LogitsProcessorList(
        [
            _ExpertLogitsProcessor(
                traj_token_offset=traj_token_start_idx,
                traj_vocab_size=traj_vocab_size,
            )
        ]
    )

    with torch.autocast(
        "cuda",
        dtype=dtype,
    ):
        vlm_outputs = vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **extra_kwargs,
        )

    rope_deltas = getattr(
        vlm.model,
        "rope_deltas",
        None,
    )

    if rope_deltas is None:
        rope_deltas = torch.zeros(
            input_ids.shape[0],
            1,
            device=input_ids.device,
            dtype=torch.long,
        )

    vlm_outputs.sequences = _replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=traj_future_start_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    return (
        vlm_outputs.past_key_values,
        rope_deltas,
        vlm_outputs.sequences,
    )


# =============================================================================
# !!! 追加: 豪雨検出時、CoTを固定文に強制差し替えてKV cacheを取得
#
# 自由生成（generate）には依存せず、
# <|cot_start|> + 固定のCoT文 + <|cot_end|> + <|traj_future_start|>
# を組み立てて1回のforwardでKV cacheを構築する。
# メッセージ内容が確率的生成に左右されないことを保証するため。
# =============================================================================

@torch.no_grad()
def _get_vlm_kv_cache_forced_cot(
    cfg: "_InferenceConfig",
    vlm: nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cot_message: str,
    cot_end_id: int,
    traj_future_start_id: int,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    device = input_ids.device
    batch_size = input_ids.shape[0]

    cot_message_ids = tokenizer(
        cot_message,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    tail_ids = torch.cat(
        [
            cot_message_ids,
            torch.tensor(
                [[cot_end_id, traj_future_start_id]],
                device=device,
                dtype=input_ids.dtype,
            ),
        ],
        dim=1,
    ).expand(batch_size, -1)

    full_input_ids = torch.cat([input_ids, tail_ids], dim=1)
    full_attention_mask = torch.cat(
        [attention_mask, torch.ones_like(tail_ids)],
        dim=1,
    )

    extra_kwargs = {}

    if pixel_values is not None:
        extra_kwargs["pixel_values"] = pixel_values.to(dtype=dtype)

    if image_grid_thw is not None:
        extra_kwargs["image_grid_thw"] = image_grid_thw

    with torch.autocast(
        "cuda",
        dtype=dtype,
    ):
        outputs = vlm(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=True,
            **extra_kwargs,
        )

    rope_deltas = getattr(
        vlm.model,
        "rope_deltas",
        None,
    )

    if rope_deltas is None:
        rope_deltas = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=torch.long,
        )

    return outputs.past_key_values, rope_deltas, full_input_ids


# =============================================================================
# Expert 用 position_ids / attention_mask 構築（210/alpamayo1_5.py 準拠）
# =============================================================================

def _build_expert_inputs(
    batch_size: int,
    n_diffusion_tokens: int,
    kv_seq_len: int,
    rope_deltas: torch.Tensor,
    generated_sequences: torch.Tensor,
    traj_future_start_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # <|traj_future_start|> の位置を特定
    traj_future_start_mask = (
        generated_sequences == traj_future_start_id
    )

    has_traj_future_start = traj_future_start_mask.any(dim=1)

    traj_future_start_positions = (
        traj_future_start_mask.int().argmax(dim=1)
    )

    last_token_positions = torch.full(
        (batch_size,),
        generated_sequences.shape[1] - 1,
        device=device,
    )

    valid_token_pos_id = torch.where(
        has_traj_future_start,
        traj_future_start_positions,
        last_token_positions,
    )

    if not has_traj_future_start.all():
        logger.warning(
            "<|traj_future_start|> が生成されませんでした"
            "最後のトークン位置を使用します"
        )

    # offset = traj_future_start_position + 1
    offset = valid_token_pos_id + 1

    # Position IDs（Qwen3-VL 3成分 RoPE）
    position_ids = torch.arange(
        n_diffusion_tokens,
        device=device,
    )

    position_ids = einops.repeat(
        position_ids,
        "l -> 3 b l",
        b=batch_size,
    ).clone()

    delta = rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    # Attention mask: offset 以降〜KV cache 末尾をマスク（EOS 後の余分 KV）
    attention_mask = torch.zeros(
        (
            batch_size,
            1,
            n_diffusion_tokens,
            kv_seq_len + n_diffusion_tokens,
        ),
        dtype=torch.float32,
        device=device,
    )

    for i in range(batch_size):
        ofs = int(offset[i].item())

        if ofs < kv_seq_len:
            attention_mask[
                i,
                :,
                :,
                ofs:kv_seq_len,
            ] = torch.finfo(attention_mask.dtype).min

    return position_ids, attention_mask


# =============================================================================
# Flow Matching サンプリング（Euler integration, alpamayo1_5.py 準拠）
# =============================================================================

@torch.no_grad()
def _flow_matching_sample(
    cfg: _InferenceConfig,
    vlm: nn.Module,
    tokenizer: AutoTokenizer,
    expert: nn.Module,
    action_in_proj_model: nn.Module,
    action_out_proj_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_diffusion_tokens: int,
    action_dims: tuple[int, ...],
    traj_future_start_id: int,
    traj_token_start_idx: int,
    traj_vocab_size: int,
    num_inference_steps: int = 10,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    use_non_causal: bool = True,
    is_heavy_rain: bool = False,  # !!! 追加: 豪雨検出フラグ
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow Matching Euler integration で連続アクションをサンプリング

    Returns:
        sampled_action: (B, 64, 2)
        generated_sequences: (B, seq_len) - CoT を含む生成トークン列
    """

    device = input_ids.device
    B = input_ids.shape[0]

    # !!! 追加: 豪雨検出時は自由生成をスキップし、
    # 固定のCoT文を強制的に差し込んでKV cacheを取得する
    if is_heavy_rain:
        past_key_values, rope_deltas, generated_sequences = _get_vlm_kv_cache_forced_cot(
            cfg=cfg,
            vlm=vlm,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cot_message=cfg.HEAVY_RAIN_COT_MESSAGE,
            cot_end_id=_state.cot_end_id,
            traj_future_start_id=traj_future_start_id,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dtype=dtype,
        )
    else:
        # 1) VLM generate → KV cache 取得
        past_key_values, rope_deltas, generated_sequences = _get_vlm_kv_cache(
            cfg=cfg,
            vlm=vlm,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            traj_future_start_id=traj_future_start_id,
            traj_token_start_idx=traj_token_start_idx,
            traj_vocab_size=traj_vocab_size,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dtype=dtype,
        )

    kv_seq_len = (
        past_key_values.get_seq_length()
        if hasattr(
            past_key_values,
            "get_seq_length",
        )
        else past_key_values[0][0].shape[2]
    )

    prefill_seq_len = kv_seq_len

    # Expert inputs
    position_ids, expert_attn_mask = _build_expert_inputs(
        batch_size=B,
        n_diffusion_tokens=n_diffusion_tokens,
        kv_seq_len=kv_seq_len,
        rope_deltas=rope_deltas,
        generated_sequences=generated_sequences,
        traj_future_start_id=traj_future_start_id,
        device=device,
    )

    # 2) Euler integration: t=0 (noise) → t=1 (data)
    x = torch.randn(
        B,
        *action_dims,
        device=device,
        dtype=dtype,
    )

    time_steps = torch.linspace(
        0.0,
        1.0,
        num_inference_steps + 1,
        device=device,
    )

    fwd_kwargs = {}

    if use_non_causal:
        fwd_kwargs["is_causal"] = False

    for i in range(num_inference_steps):
        dt_step = time_steps[i + 1] - time_steps[i]

        t_current = time_steps[i].view(1, 1, 1,).expand(B, 1, 1,).to(dtype=dtype)

        # action in proj → Expert Transformer → action out proj
        future_embeds = action_in_proj_model(
            x,
            t_current,
        )

        if future_embeds.dim() == 2:
            future_embeds = future_embeds.view(
                B,
                n_diffusion_tokens,
                -1,
            )

        expert_out = expert(
            inputs_embeds=future_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=expert_attn_mask,
            use_cache=True,
            **fwd_kwargs,
        )

        # Expert forward で追加されたトークンの KV を除去
        if hasattr(
            past_key_values,
            "crop",
        ):
            past_key_values.crop(prefill_seq_len)

        last_hidden = expert_out.last_hidden_state[
            :,
            -n_diffusion_tokens:,
        ]

        v = action_out_proj_model(last_hidden).view(
            -1,
            *action_dims,
        )

        x = x + dt_step * v

    return x, generated_sequences


# =============================================================================
# モデルロード（初回呼び出し時に一度だけ実行）
# =============================================================================

class _ModelState:
    """モデルとトークナイザの状態を保持するシングルトン"""

    def __init__(self):
        self.loaded = False
        self.vlm = None
        self.tokenizer = None
        self.processor = None
        self.expert = None
        self.action_in_proj = None
        self.action_out_proj = None
        self.action_space = None
        self.hist_traj_tokenizer = None
        self.traj_token_ids = None
        self.traj_token_start_idx = None
        self.hist_token_start_idx = None
        self.cot_end_id = None  # !!! 追加

        # !!! 追加: 天候分類モデル（豪雨検知用、frozen）
        self.weather_processor = None
        self.weather_model = None


_state = _ModelState()


def _load_models(
    cfg,
) -> None:
    """全モデルコンポーネントを初回のみロードする"""

    if _state.loaded:
        return

    # --- VLM（Stage 1 学習済み、frozen）---
    _state.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.STAGE1_CHECKPOINT_DIR,
        dtype=cfg.DTYPE,
        attn_implementation="flash_attention_2",
    )

    _state.tokenizer = AutoTokenizer.from_pretrained(
        cfg.STAGE1_CHECKPOINT_DIR,
    )

    # トークン ID マッピング
    _state.traj_token_start_idx = _state.tokenizer.convert_tokens_to_ids(
        "<|e0|>"
    )
    _state.hist_token_start_idx = (
        _state.traj_token_start_idx + cfg.FUTURE_NUM_BINS
    )

    _state.traj_token_ids = {
        k: _state.tokenizer.convert_tokens_to_ids(v)
        for k, v in TRAJ_TOKEN.items()
    }

    _state.tokenizer.traj_token_start_idx = (
        _state.traj_token_start_idx
    )
    _state.tokenizer.traj_token_ids = _state.traj_token_ids

    # !!! 追加: 豪雨CoT強制挿入で使う <|cot_end|> のID
    _state.cot_end_id = _state.tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS["cot_end"]
    )

    # プロセッサ
    _state.processor = AutoProcessor.from_pretrained(
        cfg.VLM_MODEL,
        min_pixels=cfg.MIN_PIXELS,
        max_pixels=cfg.MAX_PIXELS,
    )

    _state.processor.tokenizer = _state.tokenizer

    # VLM freeze
    _state.vlm.eval()

    for param in _state.vlm.parameters():
        param.requires_grad = False

    _state.vlm = _state.vlm.to(cfg.DEVICE)

    # --- Expert Model ---
    expert_save_path = os.path.join(
        cfg.SAVE_DIR,
        "expert",
    )

    if os.path.exists(expert_save_path):
        expert_config = AutoConfig.from_pretrained(
            expert_save_path
        )
    else:
        expert_config = copy.deepcopy(
            _state.vlm.config.text_config
        )
        expert_config.hidden_size = cfg.EXPERT_HIDDEN_SIZE
        expert_config.intermediate_size = (
            cfg.EXPERT_INTERMEDIATE_SIZE
        )
        expert_config.num_attention_heads = (
            cfg.EXPERT_NUM_ATTENTION_HEADS
        )
        expert_config.head_dim = cfg.EXPERT_HEAD_DIM
        expert_config.num_key_value_heads = min(
            getattr(
                expert_config,
                "num_key_value_heads",
                cfg.EXPERT_NUM_ATTENTION_HEADS,
            ),
            cfg.EXPERT_NUM_ATTENTION_HEADS,
        )
        expert_config._attn_implementation = "eager"

    _state.expert = AutoModel.from_config(
        expert_config
    )

    del _state.expert.embed_tokens

    _state.expert = _state.expert.to(
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )

    # --- Action Space ---
    _state.action_space = UnicycleAccelCurvatureActionSpace(
        accel_mean=cfg.ACCEL_MEAN,
        accel_std=cfg.ACCEL_STD,
        curvature_mean=cfg.CURVATURE_MEAN,
        curvature_std=cfg.CURVATURE_STD,
        accel_bounds=list(cfg.ACCEL_BOUNDS),
        curvature_bounds=list(cfg.CURVATURE_BOUNDS),
        dt=cfg.DT,
        n_waypoints=cfg.NUM_FUTURE_WAYPOINTS,
    )

    # --- Action In Proj ---
    _state.action_in_proj = PerWaypointActionInProjV2(
        in_dims=_state.action_space.get_action_space_dims(),
        out_dim=cfg.EXPERT_HIDDEN_SIZE,
        num_enc_layers=cfg.ACTION_IN_PROJ_NUM_ENC_LAYERS,
        hidden_size=cfg.ACTION_IN_PROJ_HIDDEN_SIZE,
        max_freq=cfg.ACTION_IN_PROJ_MAX_FREQ,
        num_fourier_feats=cfg.ACTION_IN_PROJ_NUM_FOURIER_FEATS,
    )

    _state.action_in_proj = _state.action_in_proj.to(
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )

    # --- Action Out Proj ---
    _state.action_out_proj = nn.Linear(
        cfg.EXPERT_HIDDEN_SIZE,
        _state.action_space.get_action_space_dims()[-1],
    )

    _state.action_out_proj = _state.action_out_proj.to(
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )

    # --- チェックポイントロード ---
    expert_ckpt = os.path.join(
        cfg.SAVE_DIR,
        "expert",
        "model.pt",
    )

    in_proj_ckpt = os.path.join(
        cfg.SAVE_DIR,
        "action_in_proj.pt",
    )

    out_proj_ckpt = os.path.join(
        cfg.SAVE_DIR,
        "action_out_proj.pt",
    )

    _state.expert.load_state_dict(
        torch.load(
            expert_ckpt,
            map_location=cfg.DEVICE,
            weights_only=True,
        )
    )

    _state.action_in_proj.load_state_dict(
        torch.load(
            in_proj_ckpt,
            map_location=cfg.DEVICE,
            weights_only=True,
        )
    )

    _state.action_out_proj.load_state_dict(
        torch.load(
            out_proj_ckpt,
            map_location=cfg.DEVICE,
            weights_only=True,
        )
    )

    _state.expert.eval()
    _state.action_in_proj.eval()
    _state.action_out_proj.eval()

    # --- 履歴トークナイザ ---
    _state.hist_traj_tokenizer = DeltaTrajectoryTokenizer(
        num_bins=cfg.HIST_NUM_BINS
    )

    # !!! 追加: 天候分類モデル（豪雨検知用）
    # VLM/Expertとは独立したfrozenモデル。KV cache・勾配には関与しない。
    _state.weather_processor = AutoImageProcessor.from_pretrained(
        cfg.WEATHER_MODEL_NAME
    )
    _state.weather_model = SiglipForImageClassification.from_pretrained(
        cfg.WEATHER_MODEL_NAME
    )
    _state.weather_model.eval()

    for param in _state.weather_model.parameters():
        param.requires_grad = False

    _state.weather_model = _state.weather_model.to(cfg.DEVICE)

    _state.loaded = True

    logger.info(
        "全モデルのロード完了"
    )


# =============================================================================
# !!! 追加: 豪雨時の速度クランプ（後処理）
#
# action_to_traj() が出力した予測軌跡(xyz)に対し、
# 各ステップの変位ベクトルを一律スケーリングして最大速度を
# max_speed_mps 以下に抑える。
#
# 進行方向（=各ステップ変位の向き）は変えず、大きさだけを縮めるため、
# 経路の形状・操舵意図は保ったまま速度のみ低下させられる。
# =============================================================================

def _clamp_trajectory_speed(
    pred_future_xyz: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    dt: float,
    max_speed_mps: float,
) -> torch.Tensor:
    """予測軌跡の速度を max_speed_mps 以下にクランプする。

    Args:
        pred_future_xyz: (1, 1, T_fut, 3) 予測将来位置（t0基準ローカル座標系）
        ego_history_xyz: (1, 1, T_hist, 3) 自車過去位置（t0基準ローカル座標系）
        dt: waypoint間隔（秒）
        max_speed_mps: 許容する最大速度（m/s）

    Returns:
        (1, 1, T_fut, 3) 速度をクランプした予測将来位置
    """
    last_hist_xyz = ego_history_xyz[:, :, -1:, :]  # (1, 1, 1, 3) - t0位置

    prev = torch.cat(
        [last_hist_xyz, pred_future_xyz[:, :, :-1, :]],
        dim=2,
    )  # (1, 1, T_fut, 3)

    diffs = pred_future_xyz - prev  # (1, 1, T_fut, 3)

    step_speed = torch.linalg.norm(diffs[..., :2], dim=-1) / dt  # (1, 1, T_fut)
    max_speed = step_speed.max()

    if max_speed <= max_speed_mps:
        return pred_future_xyz

    alpha = max_speed_mps / max_speed
    scaled_diffs = diffs * alpha

    return last_hist_xyz + torch.cumsum(scaled_diffs, dim=2)


# =============================================================================
# メイン関数
# =============================================================================

@torch.no_grad()
def predict_trajectory(
    image_frames: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    model_variant: str = "base",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    str,
]:
    """Alpamayoでカメラ画像と自車過去軌跡から将来軌跡を予測する

    Args:
        image_frames:
            shape=(N_cam, num_frames, 3, H, W), dtype=torch.uint8
            カメラ画像
            N_cam=4, num_frames=4

            カメラ順は
            [left_cross_120, front_wide_120, right_cross_120, front_tele_30]

            フレーム順は
            [t0-0.3s, t0-0.2s, t0-0.1s, t0]

        ego_history_xyz:
            torch.Tensor, shape=(1, 1, T_hist, 3), dtype=torch.float32

            t0基準ローカル座標系における自車の過去位置
            T_hist=16、範囲は[t0-1.5s, ..., t0]、10Hz

        ego_history_rot:
            torch.Tensor, shape=(1, 1, T_hist, 3, 3), dtype=torch.float32

            t0基準ローカル座標系における自車の過去姿勢

        model_variant:
            fine tuningのsuffix

    Returns:
        pred_future_xyz:
            torch.Tensor, shape=(1, 1, T_fut, 3), dtype=torch.float32

            t0基準ローカル座標系における予測将来位置
            T_fut=64、範囲は[t0+0.1s, ..., t0+6.4s]、10Hz

        pred_future_rot:
            torch.Tensor, shape=(1, 1, T_fut, 3, 3), dtype=torch.float32

            t0基準ローカル座標系における予測将来姿勢
            T_fut=64、範囲は[t0+0.1s, ..., t0+6.4s]、10Hz

        cot_text:
            str

            VLMが生成したCoT（Chain of Thought）テキスト
            <|cot_start|> と <|cot_end|> の間のテキスト
    """

    # モデルを初回のみロード
    cfg = _InferenceConfig(
        model_variant=model_variant
    )

    _load_models(cfg)

    device = cfg.DEVICE
    dtype = cfg.DTYPE

    # =========================================================================
    # 1) 画像前処理
    # =========================================================================

    # (N_cam=4, num_frames=4, 3, H, W) -> (16, 3, H, W)
    frames_flat = image_frames.flatten(
        0,
        1,
    )

    # !!! 追加: 天候分類（豪雨検知）
    # front_wide_120（N_cam index=1）の最新フレーム（t0）で判定する
    is_heavy_rain, weather_label, weather_confidence = _classify_weather(
        cfg=cfg,
        frame_uint8=image_frames[1, -1],
    )

    logger.info(
        "weather classification: label=%s, confidence=%.3f, "
        "is_heavy_rain=%s",
        weather_label,
        weather_confidence,
        is_heavy_rain,
    )

    # =========================================================================
    # 2) VLM 入力トークン構築
    #    画像、履歴軌跡、<|cot_start|>
    #    → generate() で CoT 自己回帰生成
    #    → <|traj_future_start|> で停止
    # =========================================================================

    hist_traj_placeholder = (
        "<|traj_history_start|>"
        + "<|traj_history|>" * cfg.TOKENS_PER_HISTORY_TRAJ
        + "<|traj_history_end|>"
    )

    assistant_text = "<|cot_start|>"

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a driving assistant that generates "
                        "safe and accurate actions."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": frame,
                }
                for frame in frames_flat
            ]
            + [
                {
                    "type": "text",
                    "text": (
                        f"{hist_traj_placeholder}"
                        "Output the future trajectory."
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_text,
                }
            ],
        },
    ]

    inputs = _state.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]  # (1, seq_len)
    attention_mask = inputs["attention_mask"]  # (1, seq_len)

    # =========================================================================
    # 3) 履歴軌跡トークンの挿入
    #    DeltaTrajectoryTokenizer で位置差分をトークン化し、
    #    プレースホルダを実トークンで置換
    # =========================================================================

    hist_xyz_enc = ego_history_xyz.flatten(0, 1).flatten(0, 1).unsqueeze(0)
    hist_rot_enc = ego_history_rot.flatten(0, 1).flatten(0, 1).unsqueeze(0)

    hist_idx = _state.hist_traj_tokenizer.encode(
        hist_xyz=hist_xyz_enc[:, :1],
        hist_rot=hist_rot_enc[:, :1],
        fut_xyz=hist_xyz_enc,
        fut_rot=hist_rot_enc,
    ) + _state.hist_token_start_idx

    hist_pad_id = _state.traj_token_ids["history"]  # !!

    input_ids = input_ids.to(device)
    input_ids = _replace_pad_token(
        input_ids,
        hist_idx,
        hist_pad_id,
    )

    # 4) デバイス転送
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    extra_kwargs = {}

    if "pixel_values" in inputs:
        extra_kwargs["pixel_values"] = inputs["pixel_values"].to(
            device,
            dtype=dtype,
        )

    if "image_grid_thw" in inputs:
        extra_kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(device)

    # =========================================================================
    # 5) Flow Matching サンプリング
    #    VLM generate → KV cache → Expert Euler integration
    # =========================================================================

    action_dims = _state.action_space.get_action_space_dims()  # (64, 2)
    n_diffusion_tokens = action_dims[0]  # 64
    traj_future_start_id = _state.traj_token_ids["future_start"]

    with torch.autocast(
        "cuda",
        dtype=dtype,
    ):
        sampled_action, generated_sequences = _flow_matching_sample(
            cfg=cfg,
            vlm=_state.vlm,
            tokenizer=_state.tokenizer,
            expert=_state.expert,
            action_in_proj_model=_state.action_in_proj,
            action_out_proj_model=_state.action_out_proj,
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_diffusion_tokens=n_diffusion_tokens,
            action_dims=action_dims,
            traj_future_start_id=traj_future_start_id,
            traj_token_start_idx=_state.traj_token_start_idx,
            traj_vocab_size=cfg.TRAJ_VOCAB_SIZE,
            num_inference_steps=cfg.NUM_INFERENCE_STEPS,
            dtype=dtype,
            use_non_causal=cfg.EXPERT_NON_CAUSAL_ATTENTION,
            is_heavy_rain=is_heavy_rain,  # !!! 追加
            **extra_kwargs,
        )

    # =========================================================================
    # 6) アクション → 軌跡変換
    #    UnicycleAccelCurvatureActionSpace.action_to_traj()
    # =========================================================================

    # (1, 1, T_hist, 3) → (1, T_hist, 3),
    # (1, 1, T_hist, 3, 3) → (1, T_hist, 3, 3)
    hist_xyz = ego_history_xyz.squeeze(0).squeeze(0).unsqueeze(0).to(device)
    hist_rot = ego_history_rot.squeeze(0).squeeze(0).unsqueeze(0).to(device)

    pred_future_xyz, pred_future_rot = _state.action_space.action_to_traj(
        sampled_action.float(),
        hist_xyz,
        hist_rot,
    )

    # !!! 追加: 豪雨検出時、予測軌跡の速度を V_MAX_RAIN 以下にクランプする
    # （hist_xyz/pred_future_xyzは (1, T, 3) 形状のため、一時的に (1, 1, T, 3) に揃える）
    if is_heavy_rain:
        pred_future_xyz = _clamp_trajectory_speed(
            pred_future_xyz.unsqueeze(1),
            hist_xyz.unsqueeze(1),
            dt=cfg.DT,
            max_speed_mps=cfg.V_MAX_RAIN_MPS,
        ).squeeze(1)

    # (1, 64, 3) → (1, 1, 64, 3),
    # (1, 64, 3, 3) → (1, 1, 64, 3, 3)
    pred_future_xyz = pred_future_xyz.unsqueeze(1).cpu().float()
    pred_future_rot = pred_future_rot.unsqueeze(1).cpu().float()

    # =========================================================================
    # 7) CoT テキストのデコード
    #    <|cot_start|> と <|traj_future_start|> の間のトークンをデコード
    # =========================================================================

    cot_start_id = _state.tokenizer.convert_tokens_to_ids("<|cot_start|>")
    seq = generated_sequences[0]  # (seq_len,)

    cot_start_positions = (
        seq == cot_start_id
    ).nonzero(as_tuple=True)[0]

    traj_future_start_positions = (
        seq == traj_future_start_id
    ).nonzero(as_tuple=True)[0]

    if (
        len(cot_start_positions) > 0
        and len(traj_future_start_positions) > 0
    ):
        cot_start_pos = cot_start_positions[-1].item() + 1
        cot_end_pos = traj_future_start_positions[-1].item()

        cot_token_ids = seq[
            cot_start_pos:cot_end_pos
        ]

        cot_text = _state.tokenizer.decode(
            cot_token_ids,
            skip_special_tokens=True,
        )
    else:
        cot_text = ""

    return pred_future_xyz, pred_future_rot, cot_text