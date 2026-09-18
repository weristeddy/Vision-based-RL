from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402


class _FakeDinoV2(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.config = SimpleNamespace(patch_size=14, num_register_tokens=0)
    self.embeddings = nn.Conv2d(3, 384, kernel_size=14, stride=14)

  def forward(self, images: torch.Tensor):
    patches = (images.shape[-2] // 14) * (images.shape[-1] // 14)
    return {
      "last_hidden_state": torch.zeros(
        images.shape[0], patches + 1, 384, dtype=images.dtype, device=images.device
      ),
      "pooler_output": torch.zeros(
        images.shape[0], 384, dtype=images.dtype, device=images.device
      ),
    }


class _FakeResNet50(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    self.layer1 = nn.Conv2d(64, 256, 1)
    self.layer2 = nn.Conv2d(256, 512, 1)
    self.layer3 = nn.Conv2d(512, 1024, 1)
    self.layer4 = nn.Conv2d(1024, 2048, 1)


class _FakeR3M(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.convnet = _FakeResNet50()

  def forward(self, images: torch.Tensor):
    return torch.zeros(images.shape[0], 2048, dtype=images.dtype, device=images.device)


@pytest.fixture
def _fake_backbones(monkeypatch: pytest.MonkeyPatch) -> None:
  from vbrl.vision.backbones import dinov2, r3m

  monkeypatch.setattr(dinov2, "load", _FakeDinoV2)
  monkeypatch.setattr(r3m, "load", _FakeR3M)


def _registered_vision_configs() -> dict[str, dict]:
  from mjlab.tasks.registry import list_tasks, load_rl_cfg

  import vbrl.tasks  # noqa: F401

  found: dict[str, dict] = {}
  for task_id in sorted(list_tasks()):
    actor = getattr(load_rl_cfg(task_id), "actor", None)
    if actor is None or not str(actor.class_name).startswith("vbrl.vision.model:"):
      continue
    found[task_id] = dict(actor.cnn_cfg["vision"])
  return found


def _build(task_id: str):
  from vbrl.vision.config import VisionConfig
  from vbrl.vision.registry import build_encoder

  config = VisionConfig.from_mapping(_registered_vision_configs()[task_id])
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(20240806)
    return build_encoder(config, input_dim=(224, 224))


_SNAPSHOT = json.loads(
  (Path(__file__).parent / "data" / "vision_checkpoint_layout.json").read_text()
)
_FROZEN_LAYOUT: dict[str, list[str]] = _SNAPSHOT["layout"]


def test_every_registered_architecture_is_in_the_snapshot() -> None:
  assert sorted(_registered_vision_configs()) == sorted(_FROZEN_LAYOUT)


@pytest.mark.parametrize("task_id", sorted(_FROZEN_LAYOUT))
def test_module_tree_is_unchanged(task_id: str, _fake_backbones: None) -> None:
  encoder = _build(task_id)
  observed = [
    f"{key}:{tuple(value.shape)}"
    for key, value in sorted(encoder.state_dict().items())
  ]
  assert observed == list(_FROZEN_LAYOUT[task_id])


@pytest.mark.parametrize("adapter", ("global", "linear"))
@pytest.mark.parametrize("encoder", ("dinov2_vits14", "r3m_resnet50"))
def test_pooling_prefix_tracks_the_spatial_request(
  encoder: str, adapter: str, _fake_backbones: None
) -> None:
  from vbrl.vision.config import VisionConfig
  from vbrl.vision.registry import build_encoder

  built = build_encoder(
    VisionConfig(
      encoder=encoder,
      weights="pretrained",
      train_encoder=False,
      adapter=adapter,
    ),
    input_dim=(224, 224),
  )
  assert not isinstance(built.adapter[0], nn.AdaptiveAvgPool2d)
  assert not hasattr(built, "resnet")
