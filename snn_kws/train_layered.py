"""Layered GSU-branch streaming KWS with sum-threshold fusion.

Thesis architecture «GSN-ветви»: each frequency branch is a stack of GSU layers.
This file is the training entrypoint used in the published grid search.

Best layered result in the thesis (extended 65-epoch run):
    k=2, L=2, H=256, n_fft=256, hop=64, preset p7_full_0_2_2_5_5_8
    test final accuracy 0.9257

Neuron cell (GSU) is adapted from Spiking-FullSubNet (Hao et al., MIT license).
The rest of the model, losses, and streaming KWS pipeline is original.
"""

# Sum-threshold branch fusion projection + GSN fusion_head (no cat/Linear pre-fusion).
# Based on fast_learning.py; original left unchanged.

from __future__ import annotations

import contextlib
import gc
import math
import pickle
import random
import tarfile
import time
import urllib.request
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset


SUBBAND_PRESETS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "p1_full_0_4_4_8": (
        ("fullband", 0.0, 8000.0),
        ("0_4_khz", 0.0, 4000.0),
        ("4_8_khz", 4000.0, 8000.0),
    ),
    "p2_full_0_2_2_4_4_6_6_8": (
        ("fullband", 0.0, 8000.0),
        ("0_2_khz", 0.0, 2000.0),
        ("2_4_khz", 2000.0, 4000.0),
        ("4_6_khz", 4000.0, 6000.0),
        ("6_8_khz", 6000.0, 8000.0),
    ),
    "p3_default": (
        ("fullband", 0.0, 8000.0),
        ("0_1_khz", 0.0, 1000.0),
        ("1_4_khz", 1000.0, 4000.0),
        ("4_8_khz", 4000.0, 8000.0),
    ),
    "p4_full_0_1_1_8": (
        ("fullband", 0.0, 8000.0),
        ("0_1_khz", 0.0, 1000.0),
        ("1_8_khz", 1000.0, 8000.0),
    ),
    "p5_full_0_4_4_8": (
        ("fullband", 0.0, 8000.0),
        ("0_4_khz", 0.0, 4000.0),
        ("4_8_khz", 4000.0, 8000.0),
    ),
    "p6_full_0_3_3_6_6_8": (
        ("fullband", 0.0, 8000.0),
        ("0_3_khz", 0.0, 3000.0),
        ("3_6_khz", 3000.0, 6000.0),
        ("6_8_khz", 6000.0, 8000.0),
    ),
    "p7_full_0_2_2_5_5_8": (
        ("fullband", 0.0, 8000.0),
        ("0_2_khz", 0.0, 2000.0),
        ("2_5_khz", 2000.0, 5000.0),
        ("5_8_khz", 5000.0, 8000.0),
    ),
}
DEFAULT_SUBBAND_PRESET = "p3_default"


def resolve_subband_specs(preset: str) -> tuple[tuple[str, float, float], ...]:
    if preset not in SUBBAND_PRESETS:
        valid = ", ".join(sorted(SUBBAND_PRESETS))
        raise ValueError(f"Unknown subband preset {preset!r}. Valid presets: {valid}")
    return SUBBAND_PRESETS[preset]

SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SPEECH_COMMANDS_DIRNAME = "speech_commands_v0.02"
GSC_LABEL_NAMES = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]

LABEL_NAMES = list(GSC_LABEL_NAMES)
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
DEFAULT_DATASET_CACHE_DIR = "data/google_speech_commands"
DEFAULT_DURATION_SECONDS = 1.0
PRECOMPUTE_BATCH_SIZE = 32



def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class GSCAudioConfig:
    sample_rate: int = 16000
    duration: float = 1.0
    n_fft: int = 512
    hop_length: int = 256
    cache_dir: str = DEFAULT_DATASET_CACHE_DIR

    @property
    def target_len(self) -> int:
        return int(self.sample_rate * self.duration)


@dataclass
class SpikeFusionConfig:
    sample_rate: int = 16000
    duration: float = 1.0
    n_fft: int = 256
    hop_length: int = 32
    hidden_size: int = 128
    branch_num_layers: int = 2
    fusion_hidden_size: int = 128
    fusion_num_layers: int = 1
    num_classes: int = len(GSC_LABEL_NAMES)
    batch_size: int = 600
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    epochs: int = 50
    seed: int = 7
    cache_dir: str = DEFAULT_DATASET_CACHE_DIR
    precompute_in_memory: bool = True
    target_accuracy: float = 0.95
    subband_preset: str = DEFAULT_SUBBAND_PRESET

    @property
    def target_len(self) -> int:
        return int(self.sample_rate * self.duration)

    def to_gsc_audio_config(self) -> GSCAudioConfig:
        return GSCAudioConfig(
            sample_rate=self.sample_rate,
            duration=self.duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            cache_dir=self.cache_dir,
        )


def resolve_device(device: Optional[torch.device | str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def label_to_name(label: int) -> str:
    return LABEL_NAMES[int(label)]


def _find_audio_files(root: Path) -> list[Path]:
    audio_files = sorted(root.rglob("*.wav"))
    if audio_files:
        return audio_files
    return sorted(root.rglob("*.flac"))


def load_fixed_audio(file_path: str, sample_rate: int, target_len: int) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    audio = librosa.util.fix_length(audio, size=target_len)
    return audio.astype(np.float32)


def compute_subband_indices(
    n_fft: int,
    sample_rate: int,
    preset: str = DEFAULT_SUBBAND_PRESET,
) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    out: dict[str, np.ndarray] = {}
    for name, low_hz, high_hz in resolve_subband_specs(preset):
        if name == "fullband":
            indices = np.arange(freqs.shape[0], dtype=np.int64)
        else:
            if low_hz == 0.0:
                mask = (freqs >= low_hz) & (freqs <= high_hz)
            else:
                mask = (freqs > low_hz) & (freqs <= high_hz)
            indices = np.where(mask)[0].astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"Subband {name} is empty.")
        out[name] = indices
    return out


def resolve_feature_extraction_device(precompute_in_memory: bool) -> torch.device:
    """Use CUDA only for main-process precompute; DataLoader workers stay on CPU."""
    if precompute_in_memory and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _stft_magnitude_frames(
    waveforms: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    device: torch.device,
) -> torch.Tensor:
    if waveforms.dim() == 1:
        waveforms = waveforms.unsqueeze(0)
    if waveforms.dim() != 2:
        raise ValueError(f"Expected [B, L] or [L] waveform tensor, got {tuple(waveforms.shape)}.")

    pad_end = max(0, int(n_fft) - int(hop_length))
    waveforms = F.pad(waveforms.to(device), (0, pad_end))
    window = torch.hann_window(int(n_fft), device=device, dtype=torch.float32)
    spectrum = torch.stft(
        waveforms,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return spectrum.abs().transpose(1, 2).contiguous()


def log_magnitude_frames(magnitude_frames: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(magnitude_frames, np.ndarray):
        return np.log1p(magnitude_frames).astype(np.float32, copy=False)
    return torch.log1p(magnitude_frames).to(dtype=torch.float32).cpu().numpy()


def _frame_times_sec(num_frames: int, hop_length: int, sample_rate: int) -> np.ndarray:
    frame_starts = np.arange(int(num_frames), dtype=np.float32) * float(hop_length)
    return frame_starts / float(sample_rate)


def waveform_to_streaming_frames(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
    sample_rate: int,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cpu")
    waveform_t = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    magnitudes = _stft_magnitude_frames(
        waveform_t,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
    )
    log_frames = log_magnitude_frames(magnitudes[0])
    times_sec = _frame_times_sec(log_frames.shape[0], hop_length, sample_rate)
    return log_frames, times_sec


def waveforms_batch_to_streaming_frames(
    waveforms: list[np.ndarray],
    n_fft: int,
    hop_length: int,
    sample_rate: int,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not waveforms:
        return []
    batch = torch.stack([torch.from_numpy(np.asarray(w, dtype=np.float32)) for w in waveforms])
    magnitudes = _stft_magnitude_frames(
        batch,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
    )
    log_frames_batch = torch.log1p(magnitudes).to(dtype=torch.float32)
    if device.type == "cuda":
        log_frames_batch = log_frames_batch.cpu()
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for item in log_frames_batch:
        frames = item.numpy()
        times_sec = _frame_times_sec(frames.shape[0], hop_length, sample_rate)
        outputs.append((frames, times_sec))
    return outputs


def build_context_windows_numpy(band_frames: np.ndarray, k: int) -> np.ndarray:
    if band_frames.ndim != 2:
        raise ValueError(f"Expected [T, F] array, got shape {band_frames.shape}.")
    time_steps, freq_bins = band_frames.shape
    padded = np.concatenate([np.zeros((k, freq_bins), dtype=band_frames.dtype), band_frames], axis=0)
    windows = [padded[t : t + k + 1].reshape(-1) for t in range(time_steps)]
    return np.stack(windows).astype(np.float32)


def build_context_windows_batch(band_frames: torch.Tensor, k: int) -> torch.Tensor:
    if band_frames.dim() != 3:
        raise ValueError(f"Expected [B, T, F] tensor, got shape {tuple(band_frames.shape)}.")
    if k == 0:
        return band_frames
    batch_size, time_steps, freq_bins = band_frames.shape
    padded = F.pad(band_frames, (0, 0, int(k), 0))
    windows = padded.unfold(dimension=1, size=int(k) + 1, step=1)
    return windows.permute(0, 1, 3, 2).reshape(batch_size, time_steps, (int(k) + 1) * freq_bins)


class StreamingGSCDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        config: SpikeFusionConfig,
        precompute_in_memory: Optional[bool] = None,
    ):
        self.rows = rows
        self.config = config
        self.precompute_in_memory = (
            config.precompute_in_memory if precompute_in_memory is None else precompute_in_memory
        )
        self.feature_device = resolve_feature_extraction_device(self.precompute_in_memory)
        self._cache: list[Optional[dict]] = [None] * len(rows)
        if self.precompute_in_memory:
            self._precompute_all_items()

    def __len__(self) -> int:
        return len(self.rows)

    def _precompute_all_items(self) -> None:
        total = len(self.rows)
        started = time.perf_counter()
        print(
            f"Precomputing {total} spectrograms on {self.feature_device} "
            f"(batch_size={PRECOMPUTE_BATCH_SIZE})...",
            flush=True,
        )
        for batch_start in range(0, total, PRECOMPUTE_BATCH_SIZE):
            batch_end = min(batch_start + PRECOMPUTE_BATCH_SIZE, total)
            batch_indices = list(range(batch_start, batch_end))
            waveforms = [
                load_fixed_audio(
                    self.rows[idx]["path"],
                    sample_rate=self.config.sample_rate,
                    target_len=self.config.target_len,
                )
                for idx in batch_indices
            ]
            frame_batches = waveforms_batch_to_streaming_frames(
                waveforms,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                sample_rate=self.config.sample_rate,
                device=self.feature_device,
            )
            for idx, waveform, (log_frames, times_sec) in zip(batch_indices, waveforms, frame_batches):
                row = self.rows[idx]
                self._cache[idx] = {
                    "waveform": waveform,
                    "frames": log_frames,
                    "frame_times_sec": times_sec.astype(np.float32),
                    "label": int(row["label"]),
                    "path": row["path"],
                    "label_name": label_to_name(int(row["label"])),
                }
        elapsed = time.perf_counter() - started
        print(f"Precompute finished in {elapsed:.1f}s.", flush=True)

    def _build_item(self, idx: int) -> dict:
        row = self.rows[idx]
        waveform = load_fixed_audio(
            row["path"],
            sample_rate=self.config.sample_rate,
            target_len=self.config.target_len,
        )
        log_frames, times_sec = waveform_to_streaming_frames(
            waveform=waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            sample_rate=self.config.sample_rate,
            device=self.feature_device,
        )
        return {
            "waveform": waveform,
            "frames": log_frames,
            "frame_times_sec": times_sec.astype(np.float32),
            "label": int(row["label"]),
            "path": row["path"],
            "label_name": label_to_name(int(row["label"])),
        }

    def __getitem__(self, idx: int) -> dict:
        if not self.precompute_in_memory:
            return self._build_item(idx)
        if self._cache[idx] is None:
            self._cache[idx] = self._build_item(idx)
        return self._cache[idx]


def streaming_gsc_collate(batch: list[dict]) -> dict:
    frames = torch.stack([torch.from_numpy(item["frames"]) for item in batch], dim=0)
    frame_times = torch.stack([torch.from_numpy(item["frame_times_sec"]) for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "frames": frames,
        "frame_times_sec": frame_times,
        "labels": labels,
        "paths": [item["path"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
    }


def stratified_subsample_rows(
    rows: list[dict],
    fraction: Optional[float],
    seed: Optional[int] = None,
) -> list[dict]:
    if fraction is None or fraction >= 1.0:
        return list(rows)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Expected fraction in (0, 1], got {fraction}.")
    rng = np.random.default_rng(seed)
    grouped_rows: dict[int, list[dict]] = {}
    for row in rows:
        grouped_rows.setdefault(int(row["label"]), []).append(row)
    sampled_rows = []
    for label in sorted(grouped_rows):
        label_rows = grouped_rows[label]
        order = rng.permutation(len(label_rows))
        shuffled_rows = [label_rows[idx] for idx in order]
        sample_size = min(len(shuffled_rows), max(1, int(np.floor(len(shuffled_rows) * fraction))))
        sampled_rows.extend(shuffled_rows[:sample_size])
    order = rng.permutation(len(sampled_rows))
    return [sampled_rows[idx] for idx in order]


def _read_split_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().replace('\\', '/') for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def _resolve_gsc_root(cache_root: Path) -> Optional[Path]:
    candidates = [cache_root / SPEECH_COMMANDS_DIRNAME, cache_root]
    for candidate in candidates:
        if (candidate / "validation_list.txt").exists() and (candidate / "testing_list.txt").exists():
            if _find_audio_files(candidate):
                return candidate
    return None


def _download_speech_commands_archive(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    urllib.request.urlretrieve(SPEECH_COMMANDS_URL, tmp_path)
    tmp_path.replace(archive_path)


def _extract_speech_commands_archive(archive_path: Path, root: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)


def ensure_gsc_dataset(config: GSCAudioConfig) -> Path:
    root = Path(config.cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_gsc_root(root)
    if resolved is not None:
        return resolved

    archive_path = root / f"{SPEECH_COMMANDS_DIRNAME}.tar.gz"
    max_attempts = 2
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        if not archive_path.exists():
            _download_speech_commands_archive(archive_path)

        try:
            _extract_speech_commands_archive(archive_path, root)
        except (EOFError, tarfile.ReadError, OSError) as exc:
            last_error = exc
            if archive_path.exists():
                archive_path.unlink()
            continue

        resolved = _resolve_gsc_root(root)
        if resolved is not None:
            return resolved

        if archive_path.exists():
            archive_path.unlink()

    if last_error is not None:
        raise RuntimeError(
            f"Google Speech Commands dataset extraction failed after {max_attempts} attempts. "
            f"Please check network/disk and retry."
        ) from last_error

    raise RuntimeError(
        f"Google Speech Commands dataset was not found in {root} and automatic extraction did not succeed."
    )


def infer_label_from_path(path: Path) -> int:
    class_name = path.parent.name
    if class_name not in LABEL_TO_ID:
        raise ValueError(f"Unsupported class '{class_name}' for path {path}")
    return LABEL_TO_ID[class_name]


def _infer_split(rel_path: str, validation_set: set[str], testing_set: set[str]) -> str:
    if rel_path in testing_set:
        return "test"
    if rel_path in validation_set:
        return "val"
    return "train"


def build_gsc_frame(config: GSCAudioConfig) -> list[dict]:
    root = ensure_gsc_dataset(config)  # Получаем или скачиваем и распаковываем датасет, возвращается путь к корню датасета
    validation_set = _read_split_list(root / "validation_list.txt")  # Читаем список файлов для валидационной выборки
    testing_set = _read_split_list(root / "testing_list.txt")  # Читаем список файлов для тестовой выборки

    rows = []  # Список для хранения информации по всем аудиофайлам
    for class_name in LABEL_NAMES:  # Проходим по всем классам датасета
        class_dir = root / class_name  # Путь к папке с аудиофайлами определённого класса
        if not class_dir.exists():  # Проверяем, существует ли папка класса
            raise RuntimeError(f"Expected class directory {class_dir} was not found.")  # Если нет - ошибка

        for wav_path in sorted(class_dir.glob("*.wav")):  # Для каждого wav-файла класса (отсортировано по имени)
            rel_path = wav_path.relative_to(root).as_posix()  # Получаем относительный путь файла от корня в формате POSIX
            speaker = wav_path.stem.split("_nohash_", 1)[0]  # Определяем имя спикера по имени файла
            rows.append(  # Добавляем информацию о файле в список
                {
                    "path": wav_path.as_posix(),  # Полный путь к аудиофайлу в формате POSIX
                    "label": infer_label_from_path(wav_path),  # Получаем числовую метку класса по пути файла
                    "split": _infer_split(rel_path, validation_set, testing_set),  # Определяем, к какому сплиту относится файл
                    "speaker": speaker,  # Имя спикера
                }
            )
       

    if not rows:
        raise RuntimeError(f"No .wav files were found in selected classes under {root}.")
    return rows


def select_gsc_rows(
    config: SpikeFusionConfig,
    split: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    rows = [row for row in build_gsc_frame(config.to_gsc_audio_config()) if row["split"] == split]
    if seed is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
        rows = [rows[idx] for idx in order]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _resolve_train_fraction_for_first_run(config: SpikeFusionConfig, train_fraction: Optional[float]) -> float:
    if train_fraction is None:
        return 1.0
    return float(train_fraction)


def make_gsc_dataloaders(
    config: SpikeFusionConfig,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
    test_fraction: Optional[float] = None,
    val_fraction: Optional[float] = None,
    precompute_in_memory: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    train_drop_last: bool = False,
) -> tuple[
    StreamingGSCDataset,
    StreamingGSCDataset,
    StreamingGSCDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    seed = config.seed if seed is None else seed
    effective_train_fraction = _resolve_train_fraction_for_first_run(config, train_fraction)
    config._effective_train_fraction = effective_train_fraction

    train_rows = select_gsc_rows(config, split="train", limit=train_limit, seed=seed)
    val_rows = select_gsc_rows(config, split="val", limit=val_limit, seed=seed)
    test_rows = select_gsc_rows(config, split="test", limit=test_limit, seed=seed)
    train_rows = stratified_subsample_rows(train_rows, fraction=effective_train_fraction, seed=seed)
    val_rows = stratified_subsample_rows(val_rows, fraction=val_fraction, seed=seed)
    test_rows = stratified_subsample_rows(test_rows, fraction=test_fraction, seed=seed)

    train_ds = StreamingGSCDataset(train_rows, config, precompute_in_memory=precompute_in_memory)
    val_ds = StreamingGSCDataset(val_rows, config, precompute_in_memory=precompute_in_memory)
    test_ds = StreamingGSCDataset(test_rows, config, precompute_in_memory=precompute_in_memory)

    batch_size = config.batch_size if batch_size is None else batch_size
    num_workers = int(num_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "drop_last": False,
        "collate_fn": streaming_gsc_collate,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    # CUDA graphs require a fixed batch size, so the training loader can drop the
    # last partial batch. Validation/test stay drop_last=False (run eager).
    train_loader_kwargs = dict(loader_kwargs)
    train_loader_kwargs["drop_last"] = bool(train_drop_last)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


class TriangleSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        # Store gamma as a plain Python float on ctx instead of a saved tensor.
        # The previous version read it back via params[0].item() in backward,
        # which forces a GPU->CPU sync. That sync breaks CUDA graph capture
        # (and causes a torch.compile graph break). Math is unchanged.
        ctx.save_for_backward(input)
        ctx.gamma = float(gamma)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        gamma = ctx.gamma
        surrogate = (1.0 / (gamma * gamma)) * (gamma - inp.abs()).clamp(min=0)
        return grad_output * surrogate, None


triangle_spike = TriangleSurrogate.apply
MemoryState = namedtuple("MemoryState", ["hx", "cx"])

# Whether GSULayer should torch.compile its recurrent cell. Toggled by the
# training entrypoint before the model is constructed. Compiling the cell fuses
# the many tiny per-step kernels (matmuls, gate ops, surrogate spike) and cuts
# kernel-launch overhead, which is the main reason the GPU sits at low
# utilization in the original frame-by-frame loop.
_COMPILE_CELLS = False


def set_compile_cells(enabled: bool) -> None:
    global _COMPILE_CELLS
    _COMPILE_CELLS = bool(enabled)


class GSUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, shared_weights: bool = False, bn: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_weights = shared_weights
        self.use_bn = bn
        if shared_weights:
            self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        else:
            self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -stdv, stdv)

    def expanded_weights(self):
        if self.shared_weights:
            return self.weight_ih.repeat(2, 1), self.weight_hh.repeat(2, 1)
        return self.weight_ih, self.weight_hh

    def forward(self, input: torch.Tensor, state: MemoryState):
        hx, cx = state
        weight_ih, weight_hh = self.expanded_weights()
        # Use F.linear instead of manual mm + bias so that autocast casts the
        # bias consistently with the matmul inputs. The manual form lets
        # torch.compile fuse mm + fp32-bias into a single addmm with mismatched
        # dtypes under bf16 autocast, which raises a dtype error. Math is
        # identical: F.linear(x, W, b) == x @ W.t() + b.
        gates = F.linear(input, weight_ih, self.bias_ih) + F.linear(hx, weight_hh)
        forget_gate, cell_gate = gates.chunk(2, dim=1)
        lam = torch.sigmoid(forget_gate)
        cy = lam * cx + (1.0 - lam) * cell_gate
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = triangle_spike(cy)
        return hy, MemoryState(hy, cy)


class GSULayer(nn.Module):
    def __init__(self, cell_cls, *cell_args):
        super().__init__()
        self.cell = cell_cls(*cell_args)
        if _COMPILE_CELLS:
            # Compile the bound forward method rather than wrapping the module,
            # so the cell's parameters and state_dict keys stay unchanged and
            # checkpoints remain compatible with a non-compiled reload.
            self.cell.forward = torch.compile(self.cell.forward)

    def forward(self, input_seq: torch.Tensor, state: MemoryState):
        outputs = []
        current_state = state
        for time_idx in range(input_seq.size(0)):
            out, current_state = self.cell(input_seq[time_idx], current_state)
            outputs.append(out)
        return torch.stack(outputs, dim=0), current_state


class EfficientSpikingNeuron(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        shared_weights: bool = False,
        bn: bool = False,
    ):
        super().__init__()
        layers = [GSULayer(GSUCell, input_size, hidden_size, shared_weights, bn)]
        for _ in range(num_layers - 1):
            layers.append(GSULayer(GSUCell, hidden_size, hidden_size, shared_weights, bn))
        self.layers = nn.ModuleList(layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_seq: torch.Tensor, states: list[MemoryState]):
        output = input_seq
        new_states = []
        all_layer_outputs = [input_seq]
        for layer, state in zip(self.layers, states):
            output, new_state = layer(output, state)
            new_states.append(new_state)
            all_layer_outputs.append(output)
        return output, new_states, all_layer_outputs


def project_branch_spikes(
    branch_spikes: list[torch.Tensor],
    *,
    return_float: bool = False,
) -> torch.Tensor:
    """Element-wise sum across branches, scale 1/N, threshold at 0.5.

    Uses a straight-through estimator (STE) for the threshold: the forward
    value is the hard-thresholded binary result (identical to before), but
    the backward gradient passes through ``averaged`` unchanged. This keeps
    the binary gate semantics while letting gradients flow to all branch-head
    parameters — which is required both for correct training and for
    ``torch.cuda.make_graphed_callables`` to capture the backward successfully
    (the capture internally calls ``autograd.grad`` with ``allow_unused=False``
    and fails when any parameter with ``requires_grad=True`` is disconnected
    from the output).
    """
    stacked = torch.stack(branch_spikes, dim=0).float()
    averaged = stacked.sum(dim=0) * (1.0 / stacked.shape[0])
    if return_float:
        return averaged
    thresholded = (averaged >= 0.5).to(averaged.dtype)
    # STE: forward = thresholded; backward gradient = d/d(averaged)
    return averaged + (thresholded - averaged).detach()


class StatefulSpikeHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        skip_in_proj: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.skip_in_proj = bool(skip_in_proj)
        if self.skip_in_proj:
            if input_size != hidden_size:
                raise ValueError(
                    f"skip_in_proj requires input_size == hidden_size, got {input_size} and {hidden_size}."
                )
            self.in_proj = nn.Identity()
        else:
            self.in_proj = nn.Linear(input_size, hidden_size)
        self.gsn = EfficientSpikingNeuron(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            shared_weights=False,
            bn=False,
        )

    def init_state(self, batch_size: int, device: torch.device) -> list[MemoryState]:
        return [
            MemoryState(
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def forward_step(
        self,
        x: torch.Tensor,
        state: Optional[list[MemoryState]] = None,
        recurrent_steps: int = 1,
        return_dynamics: bool = False,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        device = x.device
        if state is None:
            state = self.init_state(x.shape[0], device)
        proj = self.in_proj(x)
        seq = proj.unsqueeze(0).repeat(recurrent_steps, 1, 1)
        spikes, new_state, _ = self.gsn(seq, state)
        branch_spikes = spikes[-1]
        if not return_dynamics:
            return branch_spikes, new_state
        dynamics = {
            "projected_input": proj.detach(),
            "micro_spikes": spikes.detach(),
            "layer_spikes": torch.stack([layer_state.hx for layer_state in new_state], dim=1).detach(),
            "layer_potentials": torch.stack([layer_state.cx for layer_state in new_state], dim=1).detach(),
        }
        return branch_spikes, new_state, dynamics


class RateCodingDecoder(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.readout = nn.Linear(input_size, num_classes)

    def forward(self, spike_seq: torch.Tensor):
        if spike_seq.dim() != 3:
            raise ValueError(f"Expected [B, T, H] spike tensor, got {tuple(spike_seq.shape)}")
        cumulative = spike_seq.float().cumsum(dim=1)
        denom = torch.arange(1, spike_seq.shape[1] + 1, device=spike_seq.device, dtype=spike_seq.dtype)
        denom = denom.view(1, -1, 1)
        rates = cumulative / denom
        logits = self.readout(rates)
        return logits, rates


class StreamingSpikeFusionClassifier(nn.Module):
    def __init__(self, config: SpikeFusionConfig, k: int, p: int):
        super().__init__()
        self.config = config
        self.k = int(k)
        self.p = int(p)
        self.subband_preset = str(config.subband_preset)
        self.band_indices = {
            name: torch.tensor(indices, dtype=torch.long)
            for name, indices in compute_subband_indices(
                config.n_fft, config.sample_rate, self.subband_preset
            ).items()
        }
        self.branch_names = list(self.band_indices.keys())
        self.branch_heads = nn.ModuleDict()
        for name in self.branch_names:
            input_size = int(self.band_indices[name].numel()) * (self.k + 1)
            self.branch_heads[name] = StatefulSpikeHead(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.branch_num_layers,
            )
        self.fusion_head = StatefulSpikeHead(
            input_size=config.fusion_hidden_size,
            hidden_size=config.fusion_hidden_size,
            num_layers=config.fusion_num_layers,
            skip_in_proj=True,
        )
        self.rate_decoder = RateCodingDecoder(config.fusion_hidden_size, config.num_classes)

    def _build_branch_contexts(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        contexts = {}
        for name in self.branch_names:
            band_index = self.band_indices[name].to(frames.device)
            band_frames = frames.index_select(-1, band_index)
            contexts[name] = build_context_windows_batch(band_frames, self.k)
        return contexts

    def forward(self, frames: torch.Tensor, return_dynamics: bool = False) -> dict:
        if frames.dim() != 3:
            raise ValueError(f"Expected [B, T, F] tensor, got shape {tuple(frames.shape)}.")
        branch_contexts = self._build_branch_contexts(frames)
        branch_states: dict[str, Optional[list[MemoryState]]] = {name: None for name in self.branch_names}
        fusion_state: Optional[list[MemoryState]] = None

        branch_spike_seq = {name: [] for name in self.branch_names}
        branch_layer_spikes = {name: [] for name in self.branch_names}
        branch_layer_potentials = {name: [] for name in self.branch_names}
        fusion_spike_seq = []
        fusion_layer_spikes = []
        fusion_layer_potentials = []
        fusion_inputs = []

        for time_idx in range(frames.shape[1]):
            current_branch_spikes = []
            for name in self.branch_names:
                if return_dynamics:
                    branch_spikes, new_state, dynamics = self.branch_heads[name].forward_step(
                        branch_contexts[name][:, time_idx, :],
                        state=branch_states[name],
                        recurrent_steps=self.p,
                        return_dynamics=True,
                    )
                    branch_layer_spikes[name].append(dynamics["layer_spikes"])
                    branch_layer_potentials[name].append(dynamics["layer_potentials"])
                else:
                    branch_spikes, new_state = self.branch_heads[name].forward_step(
                        branch_contexts[name][:, time_idx, :],
                        state=branch_states[name],
                        recurrent_steps=self.p,
                        return_dynamics=False,
                    )
                branch_states[name] = new_state
                branch_spike_seq[name].append(branch_spikes)
                current_branch_spikes.append(branch_spikes)

            fusion_pre = project_branch_spikes(current_branch_spikes)
            fusion_inputs.append(project_branch_spikes(current_branch_spikes, return_float=True))
            if return_dynamics:
                fusion_spikes, fusion_state, fusion_dynamics = self.fusion_head.forward_step(
                    fusion_pre,
                    state=fusion_state,
                    recurrent_steps=1,
                    return_dynamics=True,
                )
                fusion_layer_spikes.append(fusion_dynamics["layer_spikes"])
                fusion_layer_potentials.append(fusion_dynamics["layer_potentials"])
            else:
                fusion_spikes, fusion_state = self.fusion_head.forward_step(
                    fusion_pre,
                    state=fusion_state,
                    recurrent_steps=1,
                    return_dynamics=False,
                )
            fusion_spike_seq.append(fusion_spikes)

        outputs = {
            "branch_spike_seq": {name: torch.stack(values, dim=1) for name, values in branch_spike_seq.items()},
            "fusion_spike_seq": torch.stack(fusion_spike_seq, dim=1),
            "fusion_inputs": torch.stack(fusion_inputs, dim=1),
        }
        decoder_logits, decoder_rates = self.rate_decoder(outputs["fusion_spike_seq"])
        outputs["decoder_logits"] = decoder_logits
        outputs["decoder_rates"] = decoder_rates
        if return_dynamics:
            outputs["branch_layer_spikes"] = {
                name: torch.stack(values, dim=1) for name, values in branch_layer_spikes.items()
            }
            outputs["branch_layer_potentials"] = {
                name: torch.stack(values, dim=1) for name, values in branch_layer_potentials.items()
            }
            outputs["fusion_layer_spikes"] = torch.stack(fusion_layer_spikes, dim=1)
            outputs["fusion_layer_potentials"] = torch.stack(fusion_layer_potentials, dim=1)
        return outputs

@dataclass
class LossConfig:
    warmup_steps: int = 0
    final_weight: float = 1.0
    prefix_weight: float = 0.75
    consistency_weight: float = 0.15
    spike_reg_weight: float = 1e-4
    target_spike_rate: float = 0.12


def _weighted_prefix_ce(decoder_logits: torch.Tensor, labels: torch.Tensor, start_idx: int) -> torch.Tensor:
    prefix_logits = decoder_logits[:, start_idx:, :]
    if prefix_logits.numel() == 0:
        return decoder_logits.new_tensor(0.0)
    targets = labels.unsqueeze(1).expand(-1, prefix_logits.shape[1])
    ce = F.cross_entropy(
        prefix_logits.reshape(-1, prefix_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view(prefix_logits.shape[0], prefix_logits.shape[1])
    # Больший вес ранним шагам: хотим как можно раньше войти в правильный класс.
    weights = torch.linspace(1.0, 0.35, prefix_logits.shape[1], device=decoder_logits.device)
    weights = weights / weights.sum()
    return (ce * weights.unsqueeze(0)).sum(dim=1).mean()


def _temporal_consistency_kl(decoder_logits: torch.Tensor) -> torch.Tensor:
    if decoder_logits.shape[1] < 2:
        return decoder_logits.new_tensor(0.0)
    log_probs = torch.log_softmax(decoder_logits, dim=-1)
    probs = torch.softmax(decoder_logits, dim=-1)
    forward_kl = F.kl_div(log_probs[:, :-1, :], probs[:, 1:, :].detach(), reduction="none").sum(dim=-1)
    backward_kl = F.kl_div(log_probs[:, 1:, :], probs[:, :-1, :].detach(), reduction="none").sum(dim=-1)
    return 0.5 * (forward_kl.mean() + backward_kl.mean())


def _spike_rate_penalty(outputs: dict, target_rate: float) -> torch.Tensor:
    penalties = []
    fusion_rate = outputs["fusion_spike_seq"].float().mean()
    penalties.append(torch.abs(fusion_rate - target_rate))
    for spike_seq in outputs["branch_spike_seq"].values():
        penalties.append(torch.abs(spike_seq.float().mean() - target_rate))
    return torch.stack(penalties).mean()


def streaming_prefix_tc_loss(
    outputs: dict,
    labels: torch.Tensor,
    loss_cfg: LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the streaming loss.

    Components with zero weight are skipped entirely to avoid building useless
    autograd subgraphs. Loss components are returned as detached GPU tensors;
    the caller is responsible for syncing to host only when needed (e.g. on
    logging boundaries or at epoch end), which keeps the training loop free of
    per-batch GPU->CPU synchronizations.
    """
    decoder_logits = outputs["decoder_logits"]
    zero = decoder_logits.new_tensor(0.0)

    if loss_cfg.final_weight != 0.0:
        final_logits = decoder_logits[:, -1, :]
        loss_final = F.cross_entropy(final_logits, labels)
    else:
        loss_final = zero

    if loss_cfg.prefix_weight != 0.0:
        start_idx = min(int(loss_cfg.warmup_steps), decoder_logits.shape[1] - 1)
        loss_prefix = _weighted_prefix_ce(decoder_logits, labels, start_idx)
    else:
        loss_prefix = zero

    if loss_cfg.consistency_weight != 0.0:
        loss_consistency = _temporal_consistency_kl(decoder_logits)
    else:
        loss_consistency = zero

    if loss_cfg.spike_reg_weight != 0.0:
        loss_spike = _spike_rate_penalty(outputs, loss_cfg.target_spike_rate)
    else:
        loss_spike = zero

    loss = (
        loss_cfg.final_weight * loss_final
        + loss_cfg.prefix_weight * loss_prefix
        + loss_cfg.consistency_weight * loss_consistency
        + loss_cfg.spike_reg_weight * loss_spike
    )
    return loss, {
        "loss_total": loss.detach(),
        "loss_final": loss_final.detach(),
        "loss_prefix": loss_prefix.detach(),
        "loss_consistency": loss_consistency.detach(),
        "loss_spike": loss_spike.detach(),
    }


def _mean_time_to_correct(predictions: torch.Tensor, labels: torch.Tensor, start_idx: int) -> float:
    matches = predictions[:, start_idx:] == labels.unsqueeze(1)
    if matches.numel() == 0:
        return float("nan")
    first_hits = []
    for row in matches:
        hit_positions = torch.nonzero(row, as_tuple=False)
        if hit_positions.numel() == 0:
            first_hits.append(float("nan"))
        else:
            first_hits.append(float(hit_positions[0, 0].item() + start_idx))
    valid = [value for value in first_hits if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _stability_after_first_hit(predictions: torch.Tensor, labels: torch.Tensor, start_idx: int) -> float:
    values = []
    for pred_row, label in zip(predictions, labels):
        suffix = pred_row[start_idx:]
        matches = suffix == label
        hits = torch.nonzero(matches, as_tuple=False)
        if hits.numel() == 0:
            values.append(0.0)
            continue
        first_hit = int(hits[0, 0].item())
        tail_matches = matches[first_hit:]
        values.append(float(tail_matches.float().mean().item()))
    return float(np.mean(values)) if values else float("nan")


def compute_sequence_metrics(decoder_logits: torch.Tensor, labels: torch.Tensor, warmup_steps: int) -> dict[str, float]:
    probs = torch.softmax(decoder_logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    final_predictions = predictions[:, -1]
    final_accuracy = float((final_predictions == labels).float().mean().detach().cpu())
    start_idx = min(int(warmup_steps), decoder_logits.shape[1] - 1)
    prefix_predictions = predictions[:, start_idx:]
    prefix_targets = labels.unsqueeze(1).expand_as(prefix_predictions)
    prefix_accuracy = float((prefix_predictions == prefix_targets).float().mean().detach().cpu())
    return {
        "final_accuracy": final_accuracy,
        "prefix_accuracy": prefix_accuracy,
        "time_to_correct_frame": _mean_time_to_correct(predictions.detach().cpu(), labels.detach().cpu(), start_idx),
        "stability_after_first_hit": _stability_after_first_hit(predictions.detach().cpu(), labels.detach().cpu(), start_idx),
    }


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _final_correct_count_gpu(decoder_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Number of correct final-frame predictions, kept on-device as a tensor.

    Returns a scalar tensor so it can be accumulated without forcing a
    GPU->CPU synchronization on every batch.
    """
    final_predictions = decoder_logits[:, -1, :].argmax(dim=-1)
    return (final_predictions == labels).sum()


LOSS_PART_KEYS = ("loss_total", "loss_final", "loss_prefix", "loss_consistency", "loss_spike")


def run_streaming_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    loss_cfg: LossConfig,
    grad_clip: float,
    max_batches: Optional[int] = None,
    phase: str = "epoch",
    log_every_batches: int = 25,
    started_at: Optional[float] = None,
    lightweight: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
) -> dict[str, float]:
    """Run one streaming epoch.

    When ``lightweight`` is True (used for training), only the loss components
    and a GPU-side ``final_accuracy`` are tracked. The expensive sequence
    metrics (``time_to_correct``, ``stability_after_first_hit``) and per-batch
    GPU->CPU synchronizations are skipped entirely. Loss components are
    accumulated as on-device tensors and only synced to host at logging
    boundaries and at the end of the epoch.

    When ``lightweight`` is False (used for validation/test), the full set of
    sequence metrics is computed.
    """
    is_train = optimizer is not None
    started_at = time.perf_counter() if started_at is None else started_at
    model.train(is_train)

    loss_accum = {key: torch.zeros((), device=device) for key in LOSS_PART_KEYS}
    correct_final = torch.zeros((), device=device)
    full_accum = {
        "final_accuracy": 0.0,
        "prefix_accuracy": 0.0,
        "stability_after_first_hit": 0.0,
    }
    total_examples = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    time_to_correct_values = []

    use_autocast = autocast_dtype is not None and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        non_blocking = device.type == "cuda"
        frames = batch["frames"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)

        if use_autocast:
            autocast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            outputs = model(frames, return_dynamics=False)
            loss, loss_parts = streaming_prefix_tc_loss(outputs, labels, loss_cfg)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = int(labels.shape[0])
        total_examples += batch_size
        for key in LOSS_PART_KEYS:
            loss_accum[key] += loss_parts[key].float() * batch_size

        if lightweight:
            correct_final += _final_correct_count_gpu(outputs["decoder_logits"].detach(), labels)
        else:
            metrics = compute_sequence_metrics(
                outputs["decoder_logits"].detach(), labels.detach(), warmup_steps=loss_cfg.warmup_steps
            )
            for key in ("final_accuracy", "prefix_accuracy", "stability_after_first_hit"):
                full_accum[key] += float(metrics[key]) * batch_size
            time_to_correct_values.append(metrics["time_to_correct_frame"])

        should_log = bool(log_every_batches) and (
            (batch_idx + 1) % int(log_every_batches) == 0 or (batch_idx + 1) == total_batches
        )
        if should_log:
            denom = max(total_examples, 1)
            loss_total_val = float(loss_accum["loss_total"].item()) / denom
            if lightweight:
                final_acc_val = float(correct_final.item()) / denom
                prefix_acc_str = ""
            else:
                final_acc_val = full_accum["final_accuracy"] / denom
                prefix_acc_str = f"prefix_acc={full_accum['prefix_accuracy'] / denom:.4f} "
            print(
                f"[{phase}] batch {batch_idx + 1}/{total_batches} | "
                f"examples={total_examples} "
                f"loss={loss_total_val:.4f} "
                f"final_acc={final_acc_val:.4f} "
                f"{prefix_acc_str}"
                f"elapsed={format_elapsed(time.perf_counter() - started_at)}",
                flush=True,
            )

    denom = max(total_examples, 1)
    out = {key: float(loss_accum[key].item()) / denom for key in LOSS_PART_KEYS}
    if lightweight:
        out["final_accuracy"] = float(correct_final.item()) / denom
        out["prefix_accuracy"] = float("nan")
        out["stability_after_first_hit"] = float("nan")
        out["time_to_correct_frame"] = float("nan")
    else:
        for key in ("final_accuracy", "prefix_accuracy", "stability_after_first_hit"):
            out[key] = full_accum[key] / denom
        valid_ttc = [value for value in time_to_correct_values if not np.isnan(value)]
        out["time_to_correct_frame"] = float(np.mean(valid_ttc)) if valid_ttc else float("nan")
    return out


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def default_spike_fusion_checkpoint_path(k: int, p: int) -> Path:
    return Path("artifacts/flow_spike_fusion") / f"best_spike_fusion_k{k}_p{p}.pkl"


def save_streaming_spike_fusion_checkpoint(
    model: StreamingSpikeFusionClassifier,
    checkpoint_path: str | Path,
    best_score: float,
    history: Optional[list[dict]] = None,
    loss_cfg: Optional[LossConfig] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": _clone_state_dict(model),
        "config": dict(model.config.__dict__),
        "k": int(model.k),
        "p": int(model.p),
        "best_score": float(best_score),
    }
    if history is not None:
        payload["history"] = history
    if loss_cfg is not None:
        payload["loss_config"] = dict(loss_cfg.__dict__)
    with checkpoint_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    return checkpoint_path


def load_streaming_spike_fusion_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device | str] = None,
) -> tuple[StreamingSpikeFusionClassifier, dict]:
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)
    with checkpoint_path.open("rb") as file_obj:
        checkpoint = pickle.load(file_obj)
    config = SpikeFusionConfig(**checkpoint["config"])
    model = StreamingSpikeFusionClassifier(
        config=config,
        k=int(checkpoint["k"]),
        p=int(checkpoint["p"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def collect_probe_spike_statistics(
    model: StreamingSpikeFusionClassifier,
    frames: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float | str | int]]:
    model.eval()
    with torch.inference_mode():
        outputs = model(frames.to(device, non_blocking=(device.type == "cuda")), return_dynamics=True)
    rows = []
    for name, tensor in outputs["branch_layer_spikes"].items():
        # tensor: [B, T, L, H]
        for layer_idx in range(tensor.shape[2]):
            layer_spikes = tensor[:, :, layer_idx, :].float()
            neuron_rates = layer_spikes.mean(dim=(0, 1)).detach().cpu().numpy()
            rows.append(
                {
                    "component": f"branch/{name}/layer_{layer_idx}",
                    "mean_rate": float(neuron_rates.mean()),
                    "std_rate": float(neuron_rates.std()),
                }
            )
    fusion_tensor = outputs["fusion_layer_spikes"]
    for layer_idx in range(fusion_tensor.shape[2]):
        layer_spikes = fusion_tensor[:, :, layer_idx, :].float()
        neuron_rates = layer_spikes.mean(dim=(0, 1)).detach().cpu().numpy()
        rows.append(
            {
                "component": f"fusion/layer_{layer_idx}",
                "mean_rate": float(neuron_rates.mean()),
                "std_rate": float(neuron_rates.std()),
            }
        )
    return rows


class _LogitsWrapper(nn.Module):
    """Wraps the classifier so its forward returns only ``decoder_logits``.

    ``torch.cuda.make_graphed_callables`` requires a callable whose inputs and
    outputs are tensors (no dicts / bool kwargs), so this thin wrapper exposes
    a single-tensor-in, single-tensor-out interface over the full model.
    """

    def __init__(self, model: StreamingSpikeFusionClassifier):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model(frames, return_dynamics=False)["decoder_logits"]


def _build_graphed_forward(model: StreamingSpikeFusionClassifier, sample_frames: torch.Tensor):
    """Capture the model forward (and its backward) as a CUDA graph.

    Returns a callable that replays the captured graph. Replaying eliminates the
    per-kernel launch latency and the Python overhead of the ~T-step frame loop,
    which is the actual bottleneck (the GPU is launch-bound, not compute-bound).
    The math is identical to eager execution.
    """
    wrapper = _LogitsWrapper(model)
    wrapper.train()
    graphed = torch.cuda.make_graphed_callables(wrapper, (sample_frames,))
    return graphed


def _run_graphed_train_epoch(
    model: nn.Module,
    graphed_forward,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: LossConfig,
    grad_clip: float,
    max_batches: Optional[int] = None,
    phase: str = "epoch",
    log_every_batches: int = 25,
    started_at: Optional[float] = None,
) -> dict[str, float]:
    """Training epoch that runs the forward through a captured CUDA graph.

    Only used for training; validation/test stay eager. Mirrors the lightweight
    accounting of ``run_streaming_epoch`` (GPU-side accumulation, no per-batch
    sync except at logging boundaries).
    """
    started_at = time.perf_counter() if started_at is None else started_at
    model.train(True)
    loss_accum = {key: torch.zeros((), device=device) for key in LOSS_PART_KEYS}
    correct_final = torch.zeros((), device=device)
    total_examples = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = graphed_forward(frames)
        outputs = {"decoder_logits": logits}
        loss, loss_parts = streaming_prefix_tc_loss(outputs, labels, loss_cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_examples += batch_size
        for key in LOSS_PART_KEYS:
            loss_accum[key] += loss_parts[key].float() * batch_size
        correct_final += _final_correct_count_gpu(logits.detach(), labels)

        should_log = bool(log_every_batches) and (
            (batch_idx + 1) % int(log_every_batches) == 0 or (batch_idx + 1) == total_batches
        )
        if should_log:
            denom = max(total_examples, 1)
            print(
                f"[{phase}] batch {batch_idx + 1}/{total_batches} | "
                f"examples={total_examples} "
                f"loss={float(loss_accum['loss_total'].item()) / denom:.4f} "
                f"final_acc={float(correct_final.item()) / denom:.4f} "
                f"elapsed={format_elapsed(time.perf_counter() - started_at)}",
                flush=True,
            )

    denom = max(total_examples, 1)
    out = {key: float(loss_accum[key].item()) / denom for key in LOSS_PART_KEYS}
    out["final_accuracy"] = float(correct_final.item()) / denom
    out["prefix_accuracy"] = float("nan")
    out["stability_after_first_hit"] = float("nan")
    out["time_to_correct_frame"] = float("nan")
    return out


def scale_learning_rate(
    base_lr: float,
    batch_size: int,
    ref_batch: int,
    mode: str,
) -> float:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return float(base_lr)
    if ref_batch <= 0:
        raise ValueError(f"ref_batch must be positive, got {ref_batch}.")
    ratio = float(batch_size) / float(ref_batch)
    if mode == "sqrt":
        return float(base_lr) * math.sqrt(ratio)
    if mode == "linear":
        return float(base_lr) * ratio
    raise ValueError(f"unknown lr scale mode '{mode}' (expected none, sqrt, linear).")


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = min(max(int(warmup_epochs), 0), max(total_epochs - 1, 0))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class GatedPlateauScheduler:
    """Hold LR fixed until validation accuracy crosses a gate, then ReduceLROnPlateau."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gate_accuracy: float,
        plateau_factor: float = 0.7,
        plateau_patience: int = 4,
        plateau_threshold: float = 1e-3,
        plateau_threshold_mode: str = "abs",
        min_learning_rate: float = 3e-5,
    ) -> None:
        self.optimizer = optimizer
        self.gate_accuracy = float(gate_accuracy)
        self.gate_reached = False
        self.gate_reached_epoch: Optional[int] = None
        self.lr_reduced = False
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(plateau_factor),
            patience=int(plateau_patience),
            threshold=float(plateau_threshold),
            threshold_mode=str(plateau_threshold_mode),
            min_lr=float(min_learning_rate),
        )

    def step(self, val_accuracy: float, epoch: int) -> bool:
        """Update LR after validation. Returns True if LR was reduced this step."""
        if not self.gate_reached:
            if float(val_accuracy) >= self.gate_accuracy:
                self.gate_reached = True
                self.gate_reached_epoch = int(epoch)
            return False

        prev_lr = float(self.optimizer.param_groups[0]["lr"])
        self._plateau.step(float(val_accuracy))
        new_lr = float(self.optimizer.param_groups[0]["lr"])
        reduced = new_lr < prev_lr
        if reduced:
            self.lr_reduced = True
        return reduced


class PhasedGatePlateauCosineScheduler:
    """Three-phase LR: fixed until gate1, plateau until gate2, cosine for the rest."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        fixed_gate_accuracy: float,
        plateau_until_accuracy: float,
        plateau_factor: float = 0.7,
        plateau_patience: int = 4,
        plateau_threshold: float = 1e-3,
        plateau_threshold_mode: str = "abs",
        min_learning_rate: float = 3e-5,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(int(total_epochs), 1)
        self.fixed_gate_accuracy = float(fixed_gate_accuracy)
        self.plateau_until_accuracy = float(plateau_until_accuracy)
        self.min_learning_rate = float(min_learning_rate)
        self.phase = "fixed"
        self.fixed_gate_reached = False
        self.fixed_gate_reached_epoch: Optional[int] = None
        self.plateau_until_reached = False
        self.plateau_until_reached_epoch: Optional[int] = None
        self.cosine_start_epoch: Optional[int] = None
        self.cosine_start_lr: Optional[float] = None
        self.lr_reduced = False
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(plateau_factor),
            patience=int(plateau_patience),
            threshold=float(plateau_threshold),
            threshold_mode=str(plateau_threshold_mode),
            min_lr=float(min_learning_rate),
        )

    def _set_lr(self, lr: float) -> None:
        lr_value = float(lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_value

    def _start_cosine(self, epoch: int) -> None:
        self.phase = "cosine"
        self.plateau_until_reached = True
        self.plateau_until_reached_epoch = int(epoch)
        self.cosine_start_epoch = int(epoch)
        self.cosine_start_lr = float(self.optimizer.param_groups[0]["lr"])
        self._apply_cosine(int(epoch))

    def _apply_cosine(self, epoch: int) -> None:
        if self.cosine_start_epoch is None or self.cosine_start_lr is None:
            return
        elapsed = int(epoch) - int(self.cosine_start_epoch)
        remaining = self.total_epochs - int(self.cosine_start_epoch)
        progress = min(float(elapsed) / float(max(remaining, 1)), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_learning_rate + (self.cosine_start_lr - self.min_learning_rate) * cosine
        self._set_lr(lr)

    def step(self, val_accuracy: float, epoch: int) -> bool:
        """Update LR after validation. Returns True if LR was reduced this step."""
        val = float(val_accuracy)
        epoch_idx = int(epoch)
        lr_reduced = False

        if self.phase == "fixed":
            if val >= self.plateau_until_accuracy:
                self.fixed_gate_reached = True
                self.fixed_gate_reached_epoch = epoch_idx
                self._start_cosine(epoch_idx)
            elif val >= self.fixed_gate_accuracy:
                self.fixed_gate_reached = True
                self.fixed_gate_reached_epoch = epoch_idx
                self.phase = "plateau"
            return lr_reduced

        if self.phase == "plateau":
            prev_lr = float(self.optimizer.param_groups[0]["lr"])
            self._plateau.step(val)
            new_lr = float(self.optimizer.param_groups[0]["lr"])
            if new_lr < prev_lr:
                lr_reduced = True
                self.lr_reduced = True
            if val >= self.plateau_until_accuracy:
                self._start_cosine(epoch_idx)
            return lr_reduced

        self._apply_cosine(epoch_idx)
        return lr_reduced


def skipped_eval_metrics() -> dict[str, float]:
    keys = (
        *LOSS_PART_KEYS,
        "final_accuracy",
        "prefix_accuracy",
        "stability_after_first_hit",
        "time_to_correct_frame",
    )
    return {key: float("nan") for key in keys}


def fit_streaming_spike_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: SpikeFusionConfig,
    k: int,
    p: int,
    loss_cfg: LossConfig,
    device: Optional[torch.device | str] = None,
    verbose: bool = True,
    smoke_test: bool = False,
    smoke_train_batches: int = 1,
    smoke_eval_batches: int = 1,
    checkpoint_path: Optional[str | Path] = None,
    batch_log_every: int = 25,
    collect_probe_stats: bool = True,
    autocast_dtype: Optional[torch.dtype] = None,
    cuda_graph: bool = False,
    lr_scheduler: str = "phased",
    warmup_epochs: int = 12,
    lr_gate_accuracy: float = 0.89,
    lr_plateau_until_accuracy: float = 0.92,
    plateau_factor: float = 0.7,
    plateau_patience: int = 4,
    plateau_threshold: float = 1e-3,
    plateau_threshold_mode: str = "abs",
    min_learning_rate: float = 3e-5,
) -> tuple[StreamingSpikeFusionClassifier, list[dict], list[dict]]:
    set_seed(config.seed)
    device = resolve_device(device)
    checkpoint_path = (
        default_spike_fusion_checkpoint_path(k, p)
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    model = StreamingSpikeFusionClassifier(config=config, k=k, p=p).to(device)
    # band_indices are plain (non-parameter) tensors that .to(device) above does
    # NOT move. The forward used to copy them host->device every step; that H2D
    # copy is a synchronization that aborts CUDA-graph capture. Move them once.
    model.band_indices = {name: idx.to(device) for name, idx in model.band_indices.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    epochs = 1 if smoke_test else int(config.epochs)
    cosine_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    gated_scheduler: Optional[GatedPlateauScheduler] = None
    phased_scheduler: Optional[PhasedGatePlateauCosineScheduler] = None
    lr_scheduler_name = (lr_scheduler or "none").strip().lower()
    if lr_scheduler_name == "cosine":
        cosine_scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=epochs,
            warmup_epochs=int(warmup_epochs),
        )
    elif lr_scheduler_name == "gated_plateau":
        gated_scheduler = GatedPlateauScheduler(
            optimizer=optimizer,
            gate_accuracy=float(lr_gate_accuracy),
            plateau_factor=float(plateau_factor),
            plateau_patience=int(plateau_patience),
            plateau_threshold=float(plateau_threshold),
            plateau_threshold_mode=str(plateau_threshold_mode),
            min_learning_rate=float(min_learning_rate),
        )
    elif lr_scheduler_name == "phased":
        phased_scheduler = PhasedGatePlateauCosineScheduler(
            optimizer=optimizer,
            total_epochs=epochs,
            fixed_gate_accuracy=float(lr_gate_accuracy),
            plateau_until_accuracy=float(lr_plateau_until_accuracy),
            plateau_factor=float(plateau_factor),
            plateau_patience=int(plateau_patience),
            plateau_threshold=float(plateau_threshold),
            plateau_threshold_mode=str(plateau_threshold_mode),
            min_learning_rate=float(min_learning_rate),
        )
    elif lr_scheduler_name != "none":
        raise ValueError(
            f"unknown lr_scheduler '{lr_scheduler}' "
            f"(expected phased, gated_plateau, cosine or none)."
        )
    best_state = _clone_state_dict(model)
    best_score = -float("inf")
    best_epoch = 0
    history = []
    probe_history = []
    if collect_probe_stats:
        probe_batch = next(iter(val_loader))
        probe_frames = probe_batch["frames"][: min(2, probe_batch["frames"].shape[0])]
    else:
        probe_frames = None

    # Try to capture the forward/backward as a CUDA graph. If anything goes
    # wrong (older torch, OOM during capture, unsupported op), fall back to the
    # eager training loop so the run still proceeds.
    graphed_forward = None
    if cuda_graph and device.type == "cuda":
        try:
            sample_batch = next(iter(train_loader))
            # Use blocking transfer: non_blocking=True puts the H2D copy on a
            # separate CUDA stream and returns before the data arrives. The
            # graph-capture warmup run would then touch uninitialised GPU
            # memory, which can cause spurious errors or a corrupt graph.
            sample_frames = sample_batch["frames"].to(device)
            graphed_forward = _build_graphed_forward(model, sample_frames)
            print(
                f"[cuda-graph] captured forward/backward for batch shape "
                f"{tuple(sample_frames.shape)}.",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - report and degrade gracefully
            graphed_forward = None
            # Re-enable AMP so the eager fallback is not penalised by running
            # in full fp32. bf16 has negligible precision loss for this model
            # and roughly halves memory bandwidth + compute time.
            if autocast_dtype is None:
                autocast_dtype = torch.bfloat16
            print(
                f"[cuda-graph] capture failed ({exc!r}); "
                f"falling back to eager (bf16 AMP).",
                flush=True,
            )

    training_started_at = time.perf_counter()

    for epoch_idx in range(epochs):
        if graphed_forward is not None:
            train_metrics = _run_graphed_train_epoch(
                model=model,
                graphed_forward=graphed_forward,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_cfg=loss_cfg,
                grad_clip=config.grad_clip,
                max_batches=smoke_train_batches if smoke_test else None,
                phase=f"train epoch {epoch_idx + 1}/{epochs}",
                log_every_batches=batch_log_every if verbose else 0,
                started_at=training_started_at,
            )
        else:
            train_metrics = run_streaming_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_cfg=loss_cfg,
                grad_clip=config.grad_clip,
                max_batches=smoke_train_batches if smoke_test else None,
                phase=f"train epoch {epoch_idx + 1}/{epochs}",
                log_every_batches=batch_log_every if verbose else 0,
                started_at=training_started_at,
                lightweight=True,
                autocast_dtype=autocast_dtype,
            )
        with torch.inference_mode():
            val_metrics = run_streaming_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                loss_cfg=loss_cfg,
                grad_clip=config.grad_clip,
                max_batches=smoke_eval_batches if smoke_test else None,
                phase=f"val epoch {epoch_idx + 1}/{epochs}",
                log_every_batches=batch_log_every if verbose else 0,
                started_at=training_started_at,
                lightweight=False,
                autocast_dtype=autocast_dtype,
            )
        test_metrics = skipped_eval_metrics()
        # Model selection and early stopping are driven by the validation set.
        current_score = float(val_metrics["final_accuracy"])
        improved = current_score > best_score
        if improved:
            best_score = current_score
            best_state = _clone_state_dict(model)
            best_epoch = epoch_idx + 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_reduced_this_epoch = False
        if cosine_scheduler is not None:
            cosine_scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])
        elif phased_scheduler is not None:
            lr_reduced_this_epoch = phased_scheduler.step(current_score, epoch_idx + 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
        elif gated_scheduler is not None:
            lr_reduced_this_epoch = gated_scheduler.step(current_score, epoch_idx + 1)
            current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_row = {
            "epoch": epoch_idx + 1,
            "k": k,
            "p": p,
            "learning_rate": current_lr,
            "lr_scheduler": lr_scheduler_name,
            "lr_gate_accuracy": float(lr_gate_accuracy),
            "lr_plateau_until_accuracy": float(lr_plateau_until_accuracy),
            "lr_phase": phased_scheduler.phase if phased_scheduler is not None else "",
            "lr_gate_reached": bool(
                phased_scheduler.fixed_gate_reached
                if phased_scheduler is not None
                else gated_scheduler.gate_reached if gated_scheduler is not None else False
            ),
            "lr_plateau_until_reached": bool(phased_scheduler.plateau_until_reached)
            if phased_scheduler is not None
            else False,
            "lr_reduced": lr_reduced_this_epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "stop_reason": "",
        }
        history.append(epoch_row)

        if collect_probe_stats and probe_frames is not None:
            probe_rows = collect_probe_spike_statistics(model, probe_frames, device)
            for probe_row in probe_rows:
                probe_history.append({"epoch": epoch_idx + 1, **probe_row})

        if improved:
            save_streaming_spike_fusion_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                best_score=best_score,
                history=history,
                loss_cfg=loss_cfg,
            )

        if verbose:
            lr_phase_suffix = f" phase={phased_scheduler.phase}" if phased_scheduler is not None else ""
            print(
                f"[k={k}, p={p}] epoch {epoch_idx + 1}/{epochs} | "
                f"lr={current_lr:.2e}{lr_phase_suffix} "
                f"train_loss={train_metrics['loss_total']:.4f} "
                f"train_final_acc={train_metrics['final_accuracy']:.4f} "
                f"val_loss={val_metrics['loss_total']:.4f} "
                f"val_final_acc={val_metrics['final_accuracy']:.4f} "
                f"val_prefix_acc={val_metrics['prefix_accuracy']:.4f}",
                flush=True,
            )

        if not smoke_test and current_score >= config.target_accuracy:
            history[-1]["stop_reason"] = f"target_accuracy_reached_{config.target_accuracy:.2f}"
            break

    if history and not history[-1]["stop_reason"]:
        history[-1]["stop_reason"] = "max_epochs_reached" if not smoke_test else "smoke_test_complete"
    model.load_state_dict(best_state)
    with torch.inference_mode():
        final_test_metrics = run_streaming_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            device=device,
            loss_cfg=loss_cfg,
            grad_clip=config.grad_clip,
            max_batches=smoke_eval_batches if smoke_test else None,
            phase=f"final test (best val epoch {best_epoch})",
            log_every_batches=batch_log_every if verbose else 0,
            started_at=training_started_at,
            lightweight=False,
            autocast_dtype=autocast_dtype,
        )
    for row in history:
        if int(row["epoch"]) == int(best_epoch):
            for key, value in final_test_metrics.items():
                row[f"test_{key}"] = value
            break
    if verbose:
        print(
            f"[k={k}, p={p}] final test @ best val epoch {best_epoch} | "
            f"test_final_acc={final_test_metrics['final_accuracy']:.4f} "
            f"test_prefix_acc={final_test_metrics['prefix_accuracy']:.4f}",
            flush=True,
        )
    return model, history, probe_history

import argparse
import json
import sys


def resolve_amp_dtype(amp: Optional[str]) -> Optional[torch.dtype]:
    """Map the --amp argument to an autocast dtype (or None to disable)."""
    amp = (amp or "none").strip().lower()
    if amp in ("bf16", "bfloat16"):
        return torch.bfloat16
    if amp in ("fp16", "float16", "half"):
        return torch.float16
    if amp in ("none", "off", "fp32", "float32", ""):
        return None
    raise argparse.ArgumentTypeError(f"unknown --amp value '{amp}' (expected bf16, fp16 or none)")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def optional_tee_log(log_file: Optional[str]):
    if not log_file:
        yield
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as file_obj:
        with contextlib.redirect_stdout(TeeStream(sys.stdout, file_obj)), contextlib.redirect_stderr(
            TeeStream(sys.stderr, file_obj)
        ):
            print(f"[logging] writing stdout/stderr to {log_path}", flush=True)
            yield


def active_losses(loss_cfg: LossConfig) -> list[str]:
    active = []
    if loss_cfg.final_weight != 0.0:
        active.append("final")
    if loss_cfg.prefix_weight != 0.0:
        active.append("prefix")
    if loss_cfg.consistency_weight != 0.0:
        active.append("consistency")
    if loss_cfg.spike_reg_weight != 0.0:
        active.append("spike")
    return active


def loss_slug(loss_cfg: LossConfig) -> str:
    return "loss_" + "_".join(active_losses(loss_cfg))


def make_checkpoint_path(
    artifact_root: Path,
    k: int,
    p: int,
    n_fft: int,
    hop: int,
    slug: str,
    *,
    branch_layers: int,
    hidden_size: int,
    subband_preset: str,
) -> Path:
    return (
        artifact_root
        / f"fast_k{k}_p{p}_L{branch_layers}_H{hidden_size}_{subband_preset}_nfft{n_fft}_hop{hop}_{slug}.pkl"
    )


def run_single_experiment(
    *,
    args: argparse.Namespace,
    loss_cfg: LossConfig,
    checkpoint_path: Path,
    smoke_test: bool = False,
) -> tuple[list[dict], dict]:
    k = int(args.k)
    n_fft = int(args.n_fft)
    hop_length = int(args.hop_length)
    batch_size = int(args.batch_size)
    base_learning_rate = float(args.learning_rate)
    scaled_learning_rate = scale_learning_rate(
        base_learning_rate,
        batch_size,
        int(args.lr_base_batch),
        str(args.lr_scale_mode),
    )
    hidden_size = int(args.hidden_size)
    cfg = SpikeFusionConfig(
        epochs=int(args.epochs),
        batch_size=batch_size,
        learning_rate=scaled_learning_rate,
        hidden_size=hidden_size,
        branch_num_layers=int(args.branch_layers),
        fusion_hidden_size=hidden_size,
        fusion_num_layers=int(args.fusion_layers),
        num_classes=len(LABEL_NAMES),
        target_accuracy=float(args.target_accuracy),
        n_fft=n_fft,
        hop_length=hop_length,
        duration=float(args.duration),
        cache_dir=str(args.cache_dir),
        seed=int(args.seed),
        precompute_in_memory=True,
        subband_preset=str(args.subband_preset),
    )
    set_seed(cfg.seed)
    device = resolve_device(args.device)

    cuda_graph = bool(args.cuda_graph) and device.type == "cuda"
    # CUDA graphs and torch.compile both wrap the forward; don't combine them.
    # Graphs also need a static batch (drop the partial last train batch) and a
    # static autocast state, so we disable autocast on the graph path.
    if cuda_graph:
        set_compile_cells(False)
        autocast_dtype = None
    else:
        set_compile_cells(bool(args.compile) and device.type == "cuda")
        autocast_dtype = resolve_amp_dtype(args.amp) if device.type == "cuda" else None

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_gsc_dataloaders(
        config=cfg,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        precompute_in_memory=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_drop_last=cuda_graph,
    )

    expected_workers = int(args.num_workers)
    if (
        train_loader.num_workers != expected_workers
        or val_loader.num_workers != expected_workers
        or test_loader.num_workers != expected_workers
    ):
        raise RuntimeError(
            "num_workers was not applied consistently: "
            f"train={train_loader.num_workers}, val={val_loader.num_workers}, "
            f"test={test_loader.num_workers}, expected={expected_workers}"
        )

    model, history, _probe = fit_streaming_spike_fusion(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
        k=k,
        p=int(args.p),
        loss_cfg=loss_cfg,
        device=device,
        verbose=True,
        smoke_test=smoke_test,
        smoke_train_batches=1,
        smoke_eval_batches=1,
        checkpoint_path=checkpoint_path,
        batch_log_every=int(args.batch_log_every),
        collect_probe_stats=False,
        autocast_dtype=autocast_dtype,
        cuda_graph=cuda_graph,
        lr_scheduler=str(args.lr_scheduler),
        warmup_epochs=int(args.warmup_epochs),
        lr_gate_accuracy=float(args.lr_gate_accuracy),
        lr_plateau_until_accuracy=float(args.lr_plateau_until_accuracy),
        plateau_factor=float(args.plateau_factor),
        plateau_patience=int(args.plateau_patience),
        plateau_threshold=float(args.plateau_threshold),
        plateau_threshold_mode=str(args.plateau_threshold_mode),
        min_learning_rate=float(args.min_learning_rate),
    )

    hist_df = pd.DataFrame(history)
    # Model selection is by validation final_accuracy; test is run once at the end.
    best_val_acc = float(hist_df["val_final_accuracy"].max()) if not hist_df.empty else float("nan")
    last_val_acc = float(hist_df["val_final_accuracy"].iloc[-1]) if not hist_df.empty else float("nan")
    best_epoch = int(hist_df.loc[hist_df["val_final_accuracy"].idxmax(), "epoch"]) if not hist_df.empty else -1
    test_acc_at_best = (
        float(hist_df.loc[hist_df["val_final_accuracy"].idxmax(), "test_final_accuracy"])
        if not hist_df.empty and hist_df["test_final_accuracy"].notna().any()
        else float("nan")
    )
    last_test_acc = test_acc_at_best
    summary = {
        "k": k,
        "p": int(args.p),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "batch_size": cfg.batch_size,
        "base_learning_rate": base_learning_rate,
        "scaled_learning_rate": scaled_learning_rate,
        "lr_scale_mode": str(args.lr_scale_mode),
        "lr_base_batch": int(args.lr_base_batch),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_epochs": int(args.warmup_epochs),
        "lr_gate_accuracy": float(args.lr_gate_accuracy),
        "lr_plateau_until_accuracy": float(args.lr_plateau_until_accuracy),
        "plateau_factor": float(args.plateau_factor),
        "plateau_patience": int(args.plateau_patience),
        "plateau_threshold": float(args.plateau_threshold),
        "plateau_threshold_mode": str(args.plateau_threshold_mode),
        "min_learning_rate": float(args.min_learning_rate),
        "best_val_final_accuracy": best_val_acc,
        "last_val_final_accuracy": last_val_acc,
        "test_final_accuracy_at_best_val": test_acc_at_best,
        "last_test_final_accuracy": last_test_acc,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "stop_reason": hist_df["stop_reason"].iloc[-1] if not hist_df.empty else "",
        "checkpoint_path": checkpoint_path.as_posix(),
        "active_losses": "+".join(active_losses(loss_cfg)),
        "final_weight": loss_cfg.final_weight,
        "consistency_weight": loss_cfg.consistency_weight,
        "prefix_weight": loss_cfg.prefix_weight,
        "spike_reg_weight": loss_cfg.spike_reg_weight,
        "compile_cells": (not cuda_graph) and bool(args.compile) and device.type == "cuda",
        "cuda_graph": cuda_graph,
        "amp": "none" if cuda_graph else (str(args.amp) if device.type == "cuda" else "none"),
        "train_num_workers": int(train_loader.num_workers),
        "val_num_workers": int(val_loader.num_workers),
        "test_num_workers": int(test_loader.num_workers),
        "pin_memory": bool(args.pin_memory),
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
        "num_classes": len(LABEL_NAMES),
    }

    del model, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
    gc.collect()
    if device.type == "cuda":
        # A failed/aborted CUDA-graph capture can leave the caching allocator in
        # a "capture underway" state, which makes empty_cache() raise an internal
        # assert. Don't let cleanup crash a run that otherwise finished.
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:  # noqa: BLE001
            print(f"[cleanup] torch.cuda.empty_cache() skipped: {exc!r}", flush=True)
    return history, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train streaming spike-fusion KWS with sum-threshold pre-fusion projection "
            "and GSN fusion_head."
        )
    )
    parser.add_argument("--cache-dir", default=DEFAULT_DATASET_CACHE_DIR)
    parser.add_argument("--artifact-root", default="artifacts/fast_learning_sumfusion")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu. Defaults to auto.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--batch-log-every", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--branch-layers", type=int, default=2)
    parser.add_argument("--fusion-hidden-size", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument(
        "--subband-preset",
        default=DEFAULT_SUBBAND_PRESET,
        choices=sorted(SUBBAND_PRESETS.keys()),
        help="Frequency subband layout preset.",
    )
    parser.add_argument("--final-weight", type=float, default=1.0)
    parser.add_argument("--prefix-weight", type=float, default=0.75)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument(
        "--lr-scheduler",
        default="phased",
        choices=("phased", "gated_plateau", "cosine", "none"),
        help="Learning-rate schedule: phased (fixed until gate, plateau until second gate, "
        "then cosine), gated_plateau (fixed LR until val gate, then ReduceLROnPlateau), "
        "cosine (warmup + cosine decay), or none.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=12,
        help="Linear LR warmup length in epochs (used with --lr-scheduler cosine).",
    )
    parser.add_argument(
        "--lr-gate-accuracy",
        type=float,
        default=0.89,
        help="Keep LR fixed until val_final_accuracy reaches this threshold "
        "(phase 1 end for phased; gate for gated_plateau).",
    )
    parser.add_argument(
        "--lr-plateau-until-accuracy",
        type=float,
        default=0.92,
        help="Run ReduceLROnPlateau until val_final_accuracy reaches this threshold, "
        "then switch to cosine decay (used with --lr-scheduler phased).",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.7,
        help="LR reduction factor for ReduceLROnPlateau after the gate is reached "
        "(grid search used 0.5; default 0.7 is softer).",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=4,
        help="Epochs without val improvement before LR reduction (after gate). "
        "Grid search used 2; default 4 waits longer before cutting LR.",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-3,
        help="Minimum val accuracy improvement to reset plateau patience.",
    )
    parser.add_argument(
        "--plateau-threshold-mode",
        default="abs",
        choices=("rel", "abs"),
        help="Whether plateau threshold is relative or absolute (ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5,
        help="Minimum LR for ReduceLROnPlateau after the gate is reached.",
    )
    parser.add_argument(
        "--lr-scale-mode",
        default="none",
        choices=("sqrt", "linear", "none"),
        help="Scale --learning-rate by batch size relative to --lr-base-batch.",
    )
    parser.add_argument(
        "--lr-base-batch",
        type=int,
        default=250,
        help="Reference batch size for --learning-rate before --lr-scale-mode scaling.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the forward/backward as a CUDA graph (CUDA only) to remove "
        "kernel-launch and Python-loop overhead. Drops the partial last train "
        "batch and disables autocast/compile. Falls back to eager on failure.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the recurrent cell (CUDA only). Ignored when "
        "--cuda-graph is on. Note: ineffective here because the custom spike "
        "autograd function forces graph breaks.",
    )
    parser.add_argument(
        "--amp",
        default="none",
        help="Autocast dtype on CUDA: bf16, fp16 or none (default). Ignored when "
        "--cuda-graph is on.",
    )
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append stdout/stderr to this text log file. Defaults to '<artifact-root>/train.log'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_file is None:
        args.log_file = str(Path(args.artifact_root) / "train.log")
    with optional_tee_log(args.log_file):
        run_training(args)


def run_training(args: argparse.Namespace) -> None:
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    loss_cfg = LossConfig(
        warmup_steps=0,
        final_weight=float(args.final_weight),
        prefix_weight=float(args.prefix_weight),
        consistency_weight=float(args.consistency_weight),
        spike_reg_weight=0.0,
        target_spike_rate=0.12,
    )
    slug = loss_slug(loss_cfg)
    checkpoint_path = make_checkpoint_path(
        artifact_root,
        int(args.k),
        int(args.p),
        int(args.n_fft),
        int(args.hop_length),
        slug,
        branch_layers=int(args.branch_layers),
        hidden_size=int(args.hidden_size),
        subband_preset=str(args.subband_preset),
    )
    scaled_learning_rate = scale_learning_rate(
        float(args.learning_rate),
        int(args.batch_size),
        int(args.lr_base_batch),
        str(args.lr_scale_mode),
    )

    print(
        {
            "stage": "run_config",
            "artifact_root": artifact_root.as_posix(),
            "log_file": str(args.log_file),
            "mode": "single_run",
            "model_variant": "sumfusion",
            "k": int(args.k),
            "p": int(args.p),
            "branch_layers": int(args.branch_layers),
            "hidden_size": int(args.hidden_size),
            "subband_preset": str(args.subband_preset),
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "target_accuracy": float(args.target_accuracy),
            "base_learning_rate": float(args.learning_rate),
            "scaled_learning_rate": scaled_learning_rate,
            "lr_scale_mode": str(args.lr_scale_mode),
            "lr_base_batch": int(args.lr_base_batch),
            "lr_scheduler": str(args.lr_scheduler),
            "warmup_epochs": int(args.warmup_epochs),
            "lr_gate_accuracy": float(args.lr_gate_accuracy),
            "lr_plateau_until_accuracy": float(args.lr_plateau_until_accuracy),
            "plateau_factor": float(args.plateau_factor),
            "plateau_patience": int(args.plateau_patience),
            "plateau_threshold": float(args.plateau_threshold),
            "plateau_threshold_mode": str(args.plateau_threshold_mode),
            "min_learning_rate": float(args.min_learning_rate),
            "active_losses": "+".join(active_losses(loss_cfg)),
            "cuda_graph": bool(args.cuda_graph),
            "compile": bool(args.compile),
            "amp": str(args.amp),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
        },
        flush=True,
    )

    history, summary = run_single_experiment(
        args=args,
        loss_cfg=loss_cfg,
        checkpoint_path=checkpoint_path,
        smoke_test=bool(args.smoke_test),
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(artifact_root / "history.csv", index=False)
    (artifact_root / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    best_summary = {
        "mode": "single_run",
        "model_variant": "sumfusion",
        "config": {
            "k": int(args.k),
            "p": int(args.p),
            "branch_layers": int(args.branch_layers),
            "hidden_size": int(args.hidden_size),
            "subband_preset": str(args.subband_preset),
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "target_accuracy": float(args.target_accuracy),
        },
        "summary": summary,
        "target_reached": bool(summary.get("best_val_final_accuracy", float("nan")) >= float(args.target_accuracy)),
    }
    (artifact_root / "best_summary.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    print(f"[done] best_val_final_accuracy={summary.get('best_val_final_accuracy')}", flush=True)


if __name__ == "__main__":
    main()
