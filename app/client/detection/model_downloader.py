"""
Intel OpenVINO Model Downloader
================================
Downloads the four Open Model Zoo models required by ``IntelDetectionClient``
into  ``app/client/detection/intel/<model-name>/FP32/``  the first time the
server starts with ``DETECTION_CLIENT=intel``.

Models downloaded
-----------------
1. face-detection-adas-0001
2. facial-landmarks-35-adas-0002
3. head-pose-estimation-adas-0001
4. gaze-estimation-adas-0002

Download strategy
-----------------
1. Try ``omz_downloader`` CLI (ships with ``openvino-dev``).
2. If the CLI is missing fall back to direct HTTPS download from the
   official Intel Open Model Zoo GitHub release assets.

The folder layout after download matches what ``IntelDetectionClient`` and
the reference ``testing.py`` expect::

    intel/
    └─ face-detection-adas-0001/
       └─ FP32/
          ├─ face-detection-adas-0001.xml
          └─ face-detection-adas-0001.bin
    └─ facial-landmarks-35-adas-0002/
       └─ FP32/
          ├─ facial-landmarks-35-adas-0002.xml
          └─ facial-landmarks-35-adas-0002.bin
    ...

Environment variable overrides
-------------------------------
INTEL_MODELS_DIR   Absolute path where the ``intel/`` tree is rooted.
                   Defaults to the directory that contains this file.
INTEL_PRECISION    Model precision to download.  Default: ``FP32``.
INTEL_DEVICE       Inference device (CPU / GPU / AUTO).  Default: ``CPU``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from app.utils.logger import debug_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root of the ``intel/`` tree (next to this source file by default).
_DEFAULT_MODELS_ROOT: Path = Path(__file__).parent

# OMZ base URL for direct HTTPS fallback (commit-pinned to a stable release).
_OMZ_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "models_slides_demos/models/intel"
)

# Models and the files we need for each (xml + bin).
_MODELS: Dict[str, Dict] = {
    "face-detection-adas-0001": {
        "files": ["face-detection-adas-0001.xml", "face-detection-adas-0001.bin"],
    },
    "facial-landmarks-35-adas-0002": {
        "files": [
            "facial-landmarks-35-adas-0002.xml",
            "facial-landmarks-35-adas-0002.bin",
        ],
    },
    "head-pose-estimation-adas-0001": {
        "files": [
            "head-pose-estimation-adas-0001.xml",
            "head-pose-estimation-adas-0001.bin",
        ],
    },
    "gaze-estimation-adas-0002": {
        "files": ["gaze-estimation-adas-0002.xml", "gaze-estimation-adas-0002.bin"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _models_root() -> Path:
    raw = (os.getenv("INTEL_MODELS_DIR") or "").strip()
    return Path(raw) if raw else _DEFAULT_MODELS_ROOT


def _precision() -> str:
    return (os.getenv("INTEL_PRECISION") or "FP32").strip()


def model_dir(model_name: str) -> Path:
    """Return the directory that should contain the model's .xml / .bin files."""
    return _models_root() / "intel" / model_name / _precision()


def model_xml(model_name: str) -> Path:
    return model_dir(model_name) / f"{model_name}.xml"


def _all_present() -> bool:
    """Return True if every expected .xml and .bin file already exists."""
    for name in _MODELS:
        d = model_dir(name)
        for fname in _MODELS[name]["files"]:
            if not (d / fname).exists():
                return False
    return True


# ---------------------------------------------------------------------------
# Download via omz_downloader CLI
# ---------------------------------------------------------------------------

def _omz_cli_available() -> bool:
    return shutil.which("omz_downloader") is not None


def _download_via_omz_cli(output_dir: Path, precision: str) -> bool:
    """
    Use ``omz_downloader`` to fetch all four models into *output_dir*.
    Returns True on success, False on any failure.
    """
    for name in _MODELS:
        cmd = [
            "omz_downloader",
            "--name", name,
            "--precision", precision,
            "--output_dir", str(output_dir),
            "--num_attempts", "3",
        ]
        debug_logger.info(f"[IntelModelDownloader] Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                debug_logger.warning(
                    f"[IntelModelDownloader] omz_downloader failed for {name}:\n"
                    f"{result.stderr.strip()}"
                )
                return False
            debug_logger.info(
                f"[IntelModelDownloader] omz_downloader OK for {name}"
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            debug_logger.warning(
                f"[IntelModelDownloader] omz_downloader error for {name}: {exc}"
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Direct HTTPS fallback
# ---------------------------------------------------------------------------

def _https_url(model_name: str, fname: str, precision: str) -> str:
    return f"{_OMZ_BASE_URL}/{model_name}/{precision}/{fname}"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    debug_logger.info(f"[IntelModelDownloader] Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
        debug_logger.info(f"[IntelModelDownloader] Saved → {dest}")
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _download_via_https(precision: str) -> bool:
    """
    Directly download model files over HTTPS as a fallback.
    Returns True on success, False on any failure.
    """
    for name, info in _MODELS.items():
        dest_dir = model_dir(name)
        for fname in info["files"]:
            dest = dest_dir / fname
            if dest.exists():
                debug_logger.info(
                    f"[IntelModelDownloader] Already present, skipping: {dest}"
                )
                continue
            url = _https_url(name, fname, precision)
            try:
                _download_file(url, dest)
            except RuntimeError as exc:
                debug_logger.error(str(exc))
                return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_intel_models() -> bool:
    """
    Ensure all four Intel OpenVINO IR models exist on disk.

    Called during server startup when ``DETECTION_CLIENT=intel``.

    Returns
    -------
    bool
        ``True``  — every model file is present (either pre-existing or
                    successfully downloaded).
        ``False`` — at least one file is missing after all attempts.

    The function is idempotent: if all files already exist it returns
    ``True`` immediately without any network calls.
    """
    if _all_present():
        debug_logger.info(
            "[IntelModelDownloader] All Intel models already present. "
            "Skipping download."
        )
        return True

    precision   = _precision()
    output_root = _models_root()
    debug_logger.info(
        f"[IntelModelDownloader] One or more Intel models missing. "
        f"Downloading (precision={precision}) into {output_root / 'intel'} …"
    )

    # Attempt 1 — omz_downloader CLI (preferred; resolves transitive deps).
    if _omz_cli_available():
        debug_logger.info("[IntelModelDownloader] Using omz_downloader CLI …")
        if _download_via_omz_cli(output_root, precision) and _all_present():
            debug_logger.info("[IntelModelDownloader] CLI download complete.")
            return True
        debug_logger.warning(
            "[IntelModelDownloader] CLI download incomplete. "
            "Falling back to direct HTTPS …"
        )

    # Attempt 2 — direct HTTPS.
    debug_logger.info("[IntelModelDownloader] Using direct HTTPS download …")
    if _download_via_https(precision) and _all_present():
        debug_logger.info("[IntelModelDownloader] HTTPS download complete.")
        return True

    missing = [
        str(model_xml(n)) for n in _MODELS if not model_xml(n).exists()
    ]
    debug_logger.error(
        "[IntelModelDownloader] Download failed. Missing files:\n  "
        + "\n  ".join(missing)
        + "\n\nManual install:\n"
        + "  pip install openvino-dev\n"
        + f"  omz_downloader --name face-detection-adas-0001 "
        + f"--precision {precision} --output_dir {output_root}\n"
        + "  (repeat for the other three model names)"
    )
    return False


def get_model_paths() -> Optional[Dict[str, str]]:
    """
    Return a dict mapping model key → absolute .xml path for all four models,
    or ``None`` if any file is missing.

    Keys: ``face_det``, ``landmarks``, ``head_pose``, ``gaze``.
    """
    mapping = {
        "face_det":  "face-detection-adas-0001",
        "landmarks": "facial-landmarks-35-adas-0002",
        "head_pose": "head-pose-estimation-adas-0001",
        "gaze":      "gaze-estimation-adas-0002",
    }
    paths: Dict[str, str] = {}
    for key, name in mapping.items():
        xml = model_xml(name)
        if not xml.exists():
            debug_logger.error(
                f"[IntelModelDownloader] Model file missing: {xml}"
            )
            return None
        paths[key] = str(xml)
    return paths
