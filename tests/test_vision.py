"""Capability, validation, and wire-format contracts for the vision registry.

Module *layout* -- the weight keys the 26 retained checkpoints load against --
is pinned separately in ``test_vision_checkpoint_layout.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from tensordict import TensorDict

import vbrl.vision.registry as registry
from vbrl.vision.config import VisionConfig
from vbrl.vision.model import VisionModel
from vbrl.vision.preprocessing import prepare_images
from vbrl.vision.registry import ADAPTERS, ENCODERS, build_encoder


BACKBONES = (
  "nature_cnn", "compact_vit", "dinov2_vits14",
  "r3m_resnet50", "r3m_resnet50_layer3",
)
ADAPTER_NAMES = (
  "global", "linear", "flatten", "flatten_relu", "local_grid",
  "spatial_softmax", "afa",
)


def _config(encoder: str, adapter: str, **overrides: object) -> VisionConfig:
  spec = ENCODERS[encoder]
  values: dict[str, object] = {
    "encoder": encoder,
    "weights": spec.weights,
    "train_encoder": spec.trainable,
    "adapter": adapter,
  }
  values.update(overrides)
  return VisionConfig(**values)


@pytest.fixture
def _fake_pretrained_backbones(monkeypatch: pytest.MonkeyPatch) -> None:
  from vbrl.vision.backbones import dinov2, r3m

  monkeypatch.setattr(dinov2, "load", nn.Identity)
  monkeypatch.setattr(r3m, "load", nn.Identity)


# --- the one registry table --------------------------------------------------


def test_declared_backbone_adapter_capability_matrix_is_complete() -> None:
  assert registry.list_encoders() == ("none", *BACKBONES)
  assert registry.list_adapters() == ("none", *ADAPTER_NAMES)
  assert tuple(ENCODERS) == BACKBONES
  assert tuple(ADAPTERS) == ADAPTER_NAMES

  expected = {
    "nature_cnn": (64, "scratch", True, None),
    "compact_vit": (128, "scratch", True, None),
    "dinov2_vits14": (384, "pretrained", False, 128),
    "r3m_resnet50": (2048, "pretrained", False, 256),
    # Same frozen network one stage earlier: half the channels, four times the
    # cells, so half the encode batch for twice the feature-map size.
    "r3m_resnet50_layer3": (1024, "pretrained", False, 128),
  }
  for name, (channels, weights, trainable, batch_size) in expected.items():
    backbone = registry.encoder_spec(name)
    assert backbone.name == name
    assert backbone.channels == channels
    assert backbone.weights == weights
    assert backbone.trainable is trainable
    assert backbone.default_encode_batch_size == batch_size

  for name in ADAPTER_NAMES:
    adapter = registry.adapter_spec(name)
    assert adapter.name == name
    assert adapter.feature_request in {"global", "spatial", "local_grid"}

  # Only spatial_softmax fixes its width, and it fixes it at 256.
  assert registry.adapter_spec("spatial_softmax").fixed_output_dim == 256
  assert all(
    registry.adapter_spec(name).fixed_output_dim is None
    for name in ADAPTER_NAMES
    if name != "spatial_softmax"
  )


# The grid each backbone produces from a 224x224 image. NatureCnn's trunk has
# total stride 8; CompactViT and DINOv2 tile by patch (16 and 14 pixels); R3M's
# ResNet-50 has stride 32.
NATIVE_GRIDS = {
  "nature_cnn": 24,
  "compact_vit": 14,
  "dinov2_vits14": 16,
  "r3m_resnet50": 7,
  "r3m_resnet50_layer3": 14,
}


@pytest.mark.usefixtures("_fake_pretrained_backbones")
def test_new_work_reads_the_native_grid_with_a_head_lighter_than_its_encoder() -> None:
  """The two rules behind CURRENT_ARCHITECTURES, checked against real modules.

  A local grid smaller than the encoder's own is pure loss, and it is unequal
  loss -- pooling to 7 left R3M untouched while discarding twelve of every
  thirteen NatureCnn cells. Reading the native grid instead costs
  grid^2 x projected_channels x output_dim in the adapter's dense layer, which
  is 9.4M parameters on NatureCnn's 76k trunk, so the scratch encoders use the
  pooling heads that have no dense flatten.
  """
  from vbrl.vision.architectures import ARCHITECTURES, CURRENT_ARCHITECTURES

  scratch_seen = set()
  for token in CURRENT_ARCHITECTURES:
    config = ARCHITECTURES[token]
    if config.adapter in ("local_grid", "flatten", "flatten_relu"):
      assert config.target_grid_size == NATIVE_GRIDS[config.encoder], token

    if config.weights != "scratch":
      continue
    scratch_seen.add(config.encoder)
    encoder = build_encoder(config, input_dim=(224, 224))
    backbone = sum(p.numel() for p in encoder.backbone.parameters())
    adapter = sum(p.numel() for p in encoder.adapter.parameters())
    if config.adapter in ("flatten", "flatten_relu"):
      # The deliberate exception: the plain Nature-CNN head, whose dense layer
      # would sit in the policy MLP instead if this row had no adapter at all.
      # Pinned to exactly that matrix so nothing else creeps in beside it.
      # `flatten` adds a LayerNorm (two vectors of width output_dim);
      # `flatten_relu` is ManiSkill's rectified head and adds nothing.
      channels = ENCODERS[config.encoder].channels
      flat = channels * config.target_grid_size**2
      trailing = 1 if config.adapter == "flatten_relu" else 3
      assert adapter == flat * config.output_dim + trailing * config.output_dim, token
    else:
      assert adapter < backbone, (token, adapter, backbone)

  assert scratch_seen == {"nature_cnn", "compact_vit"}


def test_afa_head_count_follows_the_published_64_wide_split() -> None:
  """``Afa<N>`` is never chosen by hand: N is the encoder's channels over 64.

  ``AttentionPoolLatent`` splits channels into 64-wide heads, which is DINOv2's
  own head width (384/6) and timm's ``num_heads = dim // 64`` default. Picking N
  freely would silently change head width per encoder and make the AFA column
  compare four different attention shapes.
  """
  from vbrl.vision.architectures import (
    AFA_HEAD_DIM,
    ARCHITECTURES,
    CURRENT_ARCHITECTURES,
  )

  seen = 0
  for token in CURRENT_ARCHITECTURES:
    config = ARCHITECTURES[token]
    if config.adapter != "afa":
      continue
    seen += 1
    channels = ENCODERS[config.encoder].channels
    assert config.afa_num_heads == channels // AFA_HEAD_DIM, token
    assert channels % config.afa_num_heads == 0, token
    assert token.endswith(f"-Afa{config.afa_num_heads}"), token

  assert seen == 5


@pytest.mark.usefixtures("_fake_pretrained_backbones")
@pytest.mark.parametrize("encoder", BACKBONES)
@pytest.mark.parametrize("adapter", ADAPTER_NAMES)
def test_every_encoder_combines_with_every_adapter_and_loads_strictly(
  encoder: str, adapter: str
) -> None:
  config = _config(encoder, adapter)
  model = build_encoder(config, input_dim=(224, 224))
  restored = build_encoder(config, input_dim=(224, 224))

  expected_dim = ENCODERS[encoder].channels if adapter == "global" else 256
  assert model.output_dim == expected_dim
  restored.load_state_dict(model.state_dict(), strict=True)


def test_linear_encoder_keeps_legacy_single_projection_layout() -> None:
  """The eight registered ``*-Linear-*`` checkpoints load against this shape."""
  model = build_encoder(_config("nature_cnn", "linear"), input_dim=(64, 64))

  assert model.output_dim == 256
  assert sum(isinstance(m, nn.Linear) for m in model.adapter.modules()) == 1
  assert not any(
    isinstance(m, (nn.ELU, nn.GELU, nn.ReLU, nn.SiLU))
    for m in model.adapter.modules()
  )


@pytest.mark.parametrize(
  ("adapter", "expected_grid"),
  (("local_grid", 7), ("spatial_softmax", 16), ("afa", 16)),
)
def test_dinov2_selects_the_adapter_feature_boundary(
  monkeypatch: pytest.MonkeyPatch, adapter: str, expected_grid: int
) -> None:
  class FakeDino(nn.Module):
    config = SimpleNamespace(patch_size=14, num_register_tokens=0)

    def forward(self, images: torch.Tensor):
      return {
        "last_hidden_state": torch.zeros(
          images.shape[0], 257, 384, dtype=images.dtype, device=images.device
        )
      }

  from vbrl.vision.backbones import dinov2

  monkeypatch.setattr(dinov2, "load", FakeDino)
  encoder = build_encoder(
    _config("dinov2_vits14", adapter, target_grid_size=7), input_dim=(224, 224)
  )

  features = encoder.encode_features(torch.zeros(2, 3, 224, 224))
  assert features.shape == (2, 384, expected_grid, expected_grid)


def test_r3m_spatial_encoder_retains_checkpoint_alias(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class FakeR3M(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.convnet = nn.Conv2d(3, 2048, 1)

  from vbrl.vision.backbones import r3m

  monkeypatch.setattr(r3m, "load", FakeR3M)
  encoder = build_encoder(
    _config("r3m_resnet50", "local_grid"), input_dim=(224, 224)
  )

  assert encoder.resnet is encoder.backbone.convnet
  assert {
    "backbone.convnet.weight",
    "backbone.convnet.bias",
    "resnet.weight",
    "resnet.bias",
  } <= set(encoder.state_dict())


# --- configuration validation ------------------------------------------------


@pytest.mark.parametrize(
  ("overrides", "message"),
  (
    ({"encode_batch_size": 0}, "encode_batch_size.*positive"),
    ({"encode_batch_size": -1}, "encode_batch_size.*positive"),
    ({"encoder_autocast_dtype": "float64"}, "autocast dtype"),
  ),
)
def test_invalid_vision_options_raise_on_validate(
  overrides: dict, message: str
) -> None:
  with pytest.raises(ValueError, match=message):
    _config("nature_cnn", "local_grid", **overrides).validate()


@pytest.mark.parametrize(
  ("num_heads", "message"),
  (
    (0, "afa_num_heads.*positive"),
    (-1, "afa_num_heads.*positive"),
    (3, r"AFA heads \(3\) must divide encoder channels \(64\)"),
  ),
)
def test_afa_head_count_must_be_positive_and_divide_channels(
  num_heads: int, message: str
) -> None:
  with pytest.raises(ValueError, match=message):
    _config("nature_cnn", "afa", afa_num_heads=num_heads).validate()


def test_spatial_softmax_rejects_noncanonical_output_dim() -> None:
  with pytest.raises(ValueError, match="fixed 256"):
    build_encoder(
      _config("nature_cnn", "spatial_softmax", output_dim=128),
      input_dim=(224, 224),
    )


def test_retired_fields_from_historical_runs_are_ignored_not_rejected() -> None:
  """W&B run configs recorded before these options were removed must still parse.

  Every retired field only ever held its default, so dropping it cannot change
  the policy that gets rebuilt -- but the key must not raise.
  """
  from vbrl.vision.config import RETIRED_FIELDS

  historical = {
    "encoder": "dinov2_vits14",
    "weights": "pretrained",
    "train_encoder": False,
    "adapter": "local_grid",
    "output_dim": 256,
    "projected_channels": 64,
    "target_grid_size": 7,
    **dict.fromkeys(RETIRED_FIELDS, None),
  }
  config = VisionConfig.from_mapping(historical)
  config.validate()

  assert config == VisionConfig(
    encoder="dinov2_vits14",
    weights="pretrained",
    train_encoder=False,
    adapter="local_grid",
    output_dim=256,
    projected_channels=64,
    target_grid_size=7,
  )
  assert not RETIRED_FIELDS & set(config.asdict())


def test_genuinely_unknown_vision_fields_still_raise() -> None:
  with pytest.raises(ValueError, match="Unknown vision configuration fields"):
    VisionConfig.from_mapping({"encoder": "nature_cnn", "not_a_field": 1})


# --- VisionModel -------------------------------------------------------------


def test_prepare_images_accepts_nhwc_uint8() -> None:
  result = prepare_images(torch.randint(0, 256, (2, 32, 48, 3), dtype=torch.uint8))

  assert result.shape == (2, 3, 32, 48)
  assert result.dtype == torch.float32
  assert 0 <= result.min() <= result.max() <= 1


def test_vision_model_joins_proprioception_and_images_with_gradients() -> None:
  observations = TensorDict(
    {
      "actor": torch.zeros(2, 4),
      "camera": torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
    },
    batch_size=[2],
  )
  model = VisionModel(
    observations,
    {"actor": ["actor", "camera"]},
    "actor",
    2,
    hidden_dims=(16,),
    cnn_cfg={"vision": _config("nature_cnn", "spatial_softmax").asdict()},
  )

  output = model(observations)
  output.square().mean().backward()

  encoder = model.cnns["camera"]
  assert output.shape == (2, 2)
  assert any(p.grad is not None for p in encoder.backbone.parameters())
  assert any(p.grad is not None for p in encoder.adapter.parameters())


def test_vision_model_constructs_directly_from_schema_v1_agent_cnn_cfg() -> None:
  """``cnn_cfg["vision"]`` is wire format inside historical W&B run configs."""
  actor = yaml.safe_load(
    """
class_name: src.vision.model:VisionModel
hidden_dims: [32]
activation: elu
obs_normalization: true
cnn_cfg:
  latent_batchnorm: false
  vision:
    encoder: nature_cnn
    weights: scratch
    train_encoder: true
    adapter: local_grid
    output_dim: 256
    projected_channels: 64
    target_grid_size: 7
    afa_num_heads: 8
    encoder_autocast: true
    encoder_autocast_dtype: bfloat16
    encode_batch_size: null
"""
  )
  observations = TensorDict(
    {
      "actor": torch.zeros(2, 4),
      "camera": torch.zeros(2, 3, 64, 64, dtype=torch.uint8),
    },
    batch_size=[2],
  )
  def build() -> VisionModel:
    return VisionModel(
      observations,
      {"actor": ["actor", "camera"]},
      "actor",
      2,
      hidden_dims=tuple(actor["hidden_dims"]),
      activation=actor["activation"],
      obs_normalization=actor["obs_normalization"],
      cnn_cfg=actor["cnn_cfg"],
    )

  model = build()
  assert tuple(model.cnns) == ("camera",)
  assert model.cnns["camera"].output_dim == 256
  assert model(observations).shape == (2, 2)
  keys = set(model.state_dict())
  assert "cnns.camera.backbone.trunk.0.weight" in keys
  assert "cnns.camera.adapter.proj.weight" in keys

  build().load_state_dict(model.state_dict(), strict=True)
