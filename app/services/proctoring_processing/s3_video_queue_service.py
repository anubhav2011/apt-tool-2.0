"""
S3 video queue helper for scheduler processing.

Flow per job:
1) Clear local queue folder
2) Download one S3 video object
3) Return local path for processing
4) Clear local queue folder after processing
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Iterable, Set

from botocore.exceptions import BotoCoreError, ClientError

from app.client.s3_client import S3ClientFactory
from app.utils.logger import debug_logger
from app.core.config import S3Config


class S3VideoQueueService:
    """Handles S3 key building, single-file download, and queue folder cleanup."""

    def __init__(
        self,
        s3_config: Optional[S3Config] = None,
        bucket_name: Optional[str] = None,
        prefix_root: Optional[str] = None,
        local_queue_dir: str = "tmp/proctoring_queue",
    ) -> None:
        s3_config = s3_config or S3Config()

        # Prefer explicit args, otherwise use the config-driven values.
        self.bucket_name = (
            (bucket_name if bucket_name is not None else s3_config.BUCKET_NAME)
            or ""
        ).strip()

        prefix_root_value = (
            prefix_root if prefix_root is not None else s3_config.PREFIX_ROOT
        )
        self.prefix_root = (prefix_root_value or "").strip().strip("/")

        # Resolve relative queue path against the repository root so it is created
        # consistently regardless of the process working directory.
        repo_root = Path(__file__).resolve().parents[3]
        queue_path = Path(local_queue_dir)
        if not queue_path.is_absolute():
            queue_path = repo_root / queue_path

        self.local_queue_dir = queue_path
        self.local_queue_dir.mkdir(parents=True, exist_ok=True)

        self._s3_client = S3ClientFactory(s3_config).get_client()

    def clear_queue_dir(self, interview_id: Optional[str] = None) -> None:
        """
        Delete files/subfolders from the local queue folder.

        If interview_id is provided, delete anything whose name starts with
        "<interview_id>_" (final .mp4, boto3 temp suffixes like .mp4.<hash>,
        partial downloads, etc.). This is safer than wiping the whole folder
        when the API and scheduler share the same directory concurrently.
        """
        if not self.local_queue_dir.exists():
            return

        prefix = f"{interview_id}_" if interview_id else None

        for child in self.local_queue_dir.iterdir():
            try:
                if prefix is not None:
                    if not child.name.startswith(prefix):
                        continue
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
            except Exception as exc:
                debug_logger.warning(f"Failed clearing queue path {child}: {exc}")

    def clear_completed_queue_files(self, generated_interview_ids: Set[str]) -> int:
        """
        Delete local queue artifacts for interviews already generated in DB.

        Matches "<interview_id>_*" (final .mp4, boto3 .mp4.<hash> temps, etc.).
        Returns number of paths deleted.
        """
        if not generated_interview_ids:
            return 0
        if not self.local_queue_dir.exists():
            return 0

        deleted = 0
        for child in self.local_queue_dir.iterdir():
            try:
                name = child.name
                if "_" not in name:
                    continue
                interview_prefix = name.split("_", 1)[0]
                if interview_prefix not in generated_interview_ids:
                    continue
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                    deleted += 1
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                    deleted += 1
            except Exception as exc:
                debug_logger.warning(f"Failed deleting completed queue file {child}: {exc}")
        return deleted

    def delete_local_file(self, local_path: str, retries: int = 5, backoff_s: float = 0.2) -> None:
        """
        Delete a single downloaded file with a small retry loop.

        On Windows, antivirus / ffmpeg / other threads may briefly keep file handles
        open, causing WinError 32 ("being used by another process").
        """
        try:
            path = Path(local_path)
        except Exception:
            return

        for attempt in range(retries):
            try:
                if path.exists():
                    path.unlink(missing_ok=True)
                return
            except Exception as exc:
                # Retry only for common Windows file-lock situations.
                winerror = getattr(exc, "winerror", None)
                msg = str(exc).lower()
                if winerror == 32 or "being used by another process" in msg:
                    time.sleep(backoff_s)
                    continue
                debug_logger.warning(f"Failed deleting local file {path}: {exc}")
                return

        debug_logger.warning(f"Failed deleting local file {path} after {retries} retries")

    def build_s3_key(self, interview_id: str, candidate_name: str) -> str:
        """
        Build key using pattern:
        data/recording-assets/ai/beam/{interview_id}/merged/{candidate_name}.mp4
        """
        safe_candidate = re.sub(r"\s+", "_", (candidate_name or "").strip())
        safe_candidate = re.sub(r"[^A-Za-z0-9_.-]", "", safe_candidate)
        if not safe_candidate:
            raise ValueError("candidate_name is empty/invalid for S3 key generation")
        return f"{self.prefix_root}/{interview_id}/merged/{safe_candidate}.mp4"

    def download_video(self, interview_id: str, candidate_name: str) -> str:
        """Download candidate video from S3 into queue folder and return local path."""
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured")

        # Remove prior attempts and S3/boto partial files for this interview only.
        self.clear_queue_dir(interview_id=interview_id)

        key = self.build_s3_key(interview_id, candidate_name)
        # Use a unique local filename per attempt to avoid WinError 32 caused by
        # overwriting an already-open file.
        local_path = self.local_queue_dir / f"{interview_id}_{uuid.uuid4().hex}.mp4"

        debug_logger.debug(
            f"[S3-QUEUE] Downloading video: s3://{self.bucket_name}/{key} -> {local_path}"
        )
        try:
            self._s3_client.download_file(self.bucket_name, key, str(local_path))
            return str(local_path)
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"Failed to download s3://{self.bucket_name}/{key}: {exc}"
            ) from exc

    def object_exists(self, interview_id: str, candidate_name: str) -> bool:
        """Return True if the merged interview video key exists in S3 (HEAD)."""
        if not self.bucket_name:
            return False
        try:
            key = self.build_s3_key(interview_id, candidate_name)
        except ValueError:
            return False
        try:
            self._s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            debug_logger.warning(
                f"S3 head_object unexpected error for interview_id={interview_id}: {exc}"
            )
            return False
        except BotoCoreError as exc:
            debug_logger.warning(
                f"S3 head_object failed for interview_id={interview_id}: {exc}"
            )
            return False
