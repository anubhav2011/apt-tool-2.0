"""
Proctoring Scheduler
====================
Implements the flow described in the scheduler diagram:

  Every 1 hour (APScheduler IntervalTrigger):
    1. Scan ai_interview_status: proctoring_generated=0 AND proctoring_retry_count < 3
    2. If NO eligible rows → IDLE (wait next cycle)
    3. If YES → dispatch worker: process_video() (at most one in-flight job per process)
    4. on_done callback fires when worker finishes:
         SUCCESS → mark_proctoring_generated=1 → trigger_immediately()
         FAILURE → increment proctoring_retry_count
                     < 3  → wait next 1-hr cycle
                    >= 3  → DEAD alert (email + Slack + DB log)
                             check if other eligible jobs exist:
                               YES → skip dead job, dispatch others
                               NO  → process dead job as last resort
                                     (no retry increment, count stays >= 3)

Source of truth: ai_interview_status.proctoring_generated / proctoring_retry_count
"""

import threading
from datetime import datetime
from typing import Optional, Callable, List, Dict, Union, Tuple
from zoneinfo import ZoneInfo
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.utils.logger import debug_logger, is_proctoring_diagnostics_enabled
from app.utils.proctoring_processing_slot import (
    release_processing_slot,
    try_claim_processing_slot,
)
from app.core.config import SCHEDULER_CONFIG
from app.core.database import get_db
from app.repositories.proctoring_repository import ProctoringRepository
from app.repositories.ai_interview_status_repository import (
    AiInterviewStatusRepository,
    MAX_RETRIES,
)
from app.services.proctoring_service import ProctoringService

# ── Tunables (see SCHEDULER_CONFIG / env: SCHEDULER_*) ─────────────────────────
JOB_ID = "proctoring_scheduler"

# ── Module-level scheduler singleton ─────────────────────────────────────────
_scheduler: Optional[BackgroundScheduler] = None

# Prevent concurrent workers for the same interview_id within this process.
_interview_locks: dict[str, threading.Lock] = {}
_interview_locks_guard = threading.Lock()

# ── Optional external alert hook ─────────────────────────────────────────────
# Assign a callable (interview_id: str) → None to receive DEAD alerts.
dead_job_alert_hook: Optional[Callable[[str], None]] = None


def _now_in_scheduler_tz() -> datetime:
    """Timezone-aware 'now' for APScheduler (matches SCHEDULER_TIMEZONE)."""
    try:
        return datetime.now(ZoneInfo(SCHEDULER_CONFIG.TIMEZONE))
    except Exception:
        return datetime.now(ZoneInfo("UTC"))


def _sched_error_kv(error: Optional[Union[BaseException, str]] = None) -> str:
    """
    Append ` | error=...` only when an error is present.
    Success paths omit the field instead of logging misleading `error=none`.
    """
    if error is None:
        return ""
    return f" | error={error}"


# ─────────────────────────────────────────────────────────────────────────────
# Alert helper
# ─────────────────────────────────────────────────────────────────────────────

def _fire_dead_alert(interview_id: str) -> None:
    """
    Fire an alert when proctoring_retry_count reaches MAX_RETRIES.
    Extend this to send email / Slack / PagerDuty as needed.
    The ai_interview_status row is left as-is (proctoring_generated=0,
    proctoring_retry_count >= MAX_RETRIES).
    """
    debug_logger.error(
        f"[SCHEDULER] DEAD JOB — "
        f"interview_id={interview_id} | "
        f"error=max_retries_reached ({MAX_RETRIES} failures). "
        f"Alert: email + Slack + DB log (row stays as-is)."
    )
    # ── DB log ────────────────────────────────────────────────────────────
    try:
        with get_db() as db:
            repo = ProctoringRepository(db)
            repo.save_event_log(
                interview_id=interview_id,
                event_type="scheduler_dead_alert",
                event_timestamp="0:00",
                event_risk="dead",
            )
    except Exception as exc:
        debug_logger.error(
            f"[SCHEDULER] Failed to write dead-alert log — "
            f"interview_id={interview_id}{_sched_error_kv(exc)}"
        )

    # ── External hook (email / Slack) ─────────────────────────────────────
    if callable(dead_job_alert_hook):
        try:
            dead_job_alert_hook(interview_id)
        except Exception as exc:
            debug_logger.error(
                f"[SCHEDULER] dead_job_alert_hook raised — "
                f"interview_id={interview_id}{_sched_error_kv(exc)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker(interview_id: str, candidate_name: str) -> Tuple[bool, Optional[str]]:
    """
    Worker: calls VideoProcessingService.process_video() for the given
    interview_id.  Returns (True, None) on success, or (False, detail) on failure
    where detail classifies persist vs code errors for INFO+/journal logging.

    In production replace the stub body with the real processing call.
    The function runs in a daemon thread so it must not touch the Flask/
    FastAPI request context.

    All log lines include `interview_id=<X>`.
    """
    debug_logger.debug(
        f"[SCHEDULER] Worker started — interview_id={interview_id}"
    )
    try:
        from app.core.config import ProctoringConfig, S3_CONFIG
        from app.services.proctoring_processing import VideoProcessingService

        from app.services.proctoring_processing.s3_video_queue_service import S3VideoQueueService

        config = ProctoringConfig()
        processor = VideoProcessingService(config)
        s3_queue = S3VideoQueueService(s3_config=S3_CONFIG)

        local_video_path: Optional[str] = None
        try:
            # Before storing a new video in tmp/proctoring_queue, delete any
            # old queue files whose reports were already generated.
            try:
                mp4_ids: List[str] = []
                for child in s3_queue.local_queue_dir.iterdir():
                    if not child.is_file():
                        continue
                    name = child.name
                    if "_" not in name:
                        continue
                    mp4_ids.append(name.split("_", 1)[0])
                if mp4_ids:
                    with get_db() as db:
                        status_repo = AiInterviewStatusRepository(db)
                        generated_ids = status_repo.get_generated_interview_ids(
                            list({str(x) for x in mp4_ids if x})
                        )
                    deleted = s3_queue.clear_completed_queue_files(generated_ids)
                    if deleted and is_proctoring_diagnostics_enabled():
                        debug_logger.debug(
                            f"[SCHEDULER] Cleared completed queue files — "
                            f"interview_id={interview_id} | deleted={deleted}"
                        )
            except Exception as exc:
                debug_logger.warning(
                    f"[SCHEDULER] Failed clearing completed queue files — "
                    f"interview_id={interview_id}{_sched_error_kv(exc)}"
                )

            # download_video clears this interview_id_* artifacts before saving.
            local_video_path = s3_queue.download_video(interview_id, candidate_name)
            debug_logger.debug(
                f"[SCHEDULER] S3 download complete — "
                f"interview_id={interview_id} | local_file={local_video_path}"
            )
            debug_logger.debug(
                f"[SCHEDULER] Video analysis START — interview_id={interview_id}"
            )
            report = processor.process_video(local_video_path, interview_id)
            debug_logger.debug(
                f"[SCHEDULER] Video analysis COMPLETE — interview_id={interview_id}"
            )
        finally:
            # Release processing resources first, then try deleting the local file.
            processor.cleanup()
            if local_video_path:
                s3_queue.delete_local_file(local_video_path)

        # Persist report / events / summaries (same as API path via ProctoringService).
        try:
            debug_logger.debug(
                f"[SCHEDULER] DB persist START — interview_id={interview_id}"
            )
            with get_db() as db:
                proctoring_repo = ProctoringRepository(db)
                status_repo = AiInterviewStatusRepository(db)
                proctoring_service = ProctoringService(proctoring_repo, status_repo)
                proctoring_service.persist_processed_report(
                    interview_id,
                    report,
                    source_video_url=None,
                )
            debug_logger.debug(
                f"[SCHEDULER] DB persist COMPLETE — interview_id={interview_id}"
            )
            # DB persist succeeded → clear queue artifacts for this interview.
            s3_queue.clear_queue_dir(interview_id=interview_id)
        except Exception as exc:
            detail = f"persist_failed: {type(exc).__name__}: {exc}"
            debug_logger.error(
                f"[SCHEDULER] Failed to persist proctoring DB rows — "
                f"interview_id={interview_id}{_sched_error_kv(exc)}",
                exc_info=True,
            )
            return False, detail

        debug_logger.debug(
            f"[SCHEDULER] Worker finished — interview_id={interview_id}"
        )
        return True, None

    except Exception as exc:
        detail = f"code_error: {type(exc).__name__}: {exc}"
        debug_logger.error(
            f"[SCHEDULER] Worker exception — "
            f"interview_id={interview_id}{_sched_error_kv(exc)}",
            exc_info=True,
        )
        return False, detail


# ─────────────────────────────────────────────────────────────────────────────
# on_done callback (called after each worker finishes)
# ─────────────────────────────────────────────────────────────────────────────

def _on_done(
    interview_id: str,
    success: bool,
    failure_detail: Optional[str] = None,
) -> None:
    """
    Callback that runs immediately after a worker completes.

    failure_detail is set when success is False so automatic retry / dead-job
    paths log the same cause at WARNING/ERROR (visible with LOG_LEVEL=INFO).

    All log lines include `interview_id=<X>`.
    """
    if success:
        with get_db() as db:
            status_repo = AiInterviewStatusRepository(db)
            status_repo.mark_proctoring_generated(interview_id)
        debug_logger.debug(
            f"[SCHEDULER] on_done: proctoring_generated=1 — "
            f"interview_id={interview_id}"
        )
        _trigger_immediately()

    else:
        with get_db() as db:
            status_repo   = AiInterviewStatusRepository(db)
            new_retries = status_repo.increment_proctoring_retry(interview_id)

        detail_kv = _sched_error_kv(failure_detail) if failure_detail else ""
        _interval = SCHEDULER_CONFIG.INTERVAL_SECONDS
        if new_retries < MAX_RETRIES:
            debug_logger.warning(
                f"[SCHEDULER] on_done: processing failed (automatic retry) — "
                f"interview_id={interview_id}{detail_kv} | "
                f"retry={new_retries}/{MAX_RETRIES} | next_tick_within={_interval}s"
            )
        else:
            debug_logger.error(
                f"[SCHEDULER] on_done: max retries reached — "
                f"interview_id={interview_id}{detail_kv}"
            )
            _fire_dead_alert(interview_id)

            with get_db() as db:
                status_repo = AiInterviewStatusRepository(db)
                eligible_jobs = status_repo.get_pending_with_candidate_name(limit=1)

            if eligible_jobs:
                debug_logger.debug(
                    f"[SCHEDULER] Skipping dead row, dispatching other eligible jobs — "
                    f"interview_id={interview_id}"
                )
                _dispatch_jobs(eligible_jobs)
            else:
                # No other eligible jobs → last resort: re-dispatch dead row
                debug_logger.warning(
                    f"[SCHEDULER] No other eligible jobs, dispatching dead row as last resort — "
                    f"interview_id={interview_id} | "
                    f"error=no_eligible_jobs_last_resort "
                    f"(no retry increment)"
                )
                _dispatch_single(
                    interview_id,
                    candidate_name="last_resort_unknown",
                    increment_on_fail=False,
                    bypass_interview_lock=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_single(
    interview_id: str,
    candidate_name: str,
    increment_on_fail: bool = True,
    bypass_interview_lock: bool = False,
) -> None:
    """
    Spawn a daemon worker thread for one interview.

    Dispatch a worker thread for one interview.
    """
    debug_logger.debug(
        f"[SCHEDULER] Dispatching — interview_id={interview_id}"
    )

    lock: Optional[threading.Lock] = None
    if not bypass_interview_lock:
        # Acquire a per-interview lock to avoid parallel processing of the same video.
        with _interview_locks_guard:
            lock = _interview_locks.get(interview_id)
            if lock is None:
                lock = threading.Lock()
                _interview_locks[interview_id] = lock

        if not lock.acquire(blocking=False):
            debug_logger.warning(
                f"[SCHEDULER] Interview already being processed — "
                f"interview_id={interview_id} | skipping dispatch"
            )
            return

    if not try_claim_processing_slot():
        debug_logger.warning(
            f"[SCHEDULER] Global processing slot busy — "
            f"interview_id={interview_id} | skipping dispatch"
        )
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass
        return

    def _run():
        try:
            success, failure_detail = _run_worker(interview_id, candidate_name)

            if not success and not increment_on_fail:
                debug_logger.warning(
                    f"[SCHEDULER] Last-resort worker failed — "
                    f"interview_id={interview_id}{_sched_error_kv(failure_detail)} | "
                    f"error=last_resort_failed (no retry increment)"
                )
                return
            _on_done(interview_id, success, failure_detail)
        finally:
            release_processing_slot()
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True,
                         name=f"proctoring-worker-{interview_id[:8]}")
    try:
        t.start()
    except Exception as exc:
        release_processing_slot()
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass
        debug_logger.error(
            f"[SCHEDULER] Failed to start worker thread — "
            f"interview_id={interview_id}{_sched_error_kv(exc)}",
            exc_info=True,
        )
        return

    debug_logger.debug(
        f"[SCHEDULER] Worker thread started — "
        f"interview_id={interview_id} | thread={t.name}"
    )


def _dispatch_jobs(rows: List[Dict[str, str]]) -> None:
    """Dispatch eligible rows, one thread per interview."""
    for row in rows:
        _dispatch_single(row["interview_id"], row["candidate_name"])


# ─────────────────────────────────────────────────────────────────────────────
# trigger_immediately — re-fire the scheduler job without waiting 1 hour
# ─────────────────────────────────────────────────────────────────────────────

def _trigger_immediately() -> None:
    """
    Ask APScheduler to run the scheduler tick immediately (in addition to
    the normal 1-hr cycle) so newly completed jobs unblock the next batch
    without delay.
    """
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.modify_job(JOB_ID, next_run_time=_now_in_scheduler_tz())
        debug_logger.debug(
            "[SCHEDULER] trigger_immediately — "
            "interview_id=N/A | next_run=NOW"
        )
    except Exception as exc:
        debug_logger.error(
            f"[SCHEDULER] trigger_immediately failed — "
            f"interview_id=N/A{_sched_error_kv(exc)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main scheduler tick  (runs every SCHEDULER_CONFIG.INTERVAL_SECONDS)
# ─────────────────────────────────────────────────────────────────────────────

def _scheduler_tick() -> None:
    """
    One execution of the scheduler loop.

    Step 1 — Scan ai_interview_status: proctoring_generated=0 AND
             proctoring_retry_count < MAX_RETRIES
    Step 2 — If no eligible rows → IDLE (APScheduler will fire again in 1 hr)
    Step 3 — Otherwise dispatch all eligible rows as worker threads
    """
    # debug_logger.info(
    #     "[SCHEDULER] Tick — interview_id=N/A | scanning for eligible jobs"
    # )

    try:
        with get_db() as db:
            status_repo = AiInterviewStatusRepository(db)
            # Dispatch one interview per scheduler tick to prevent
            # concurrent processing of multiple interview_ids.
            eligible_jobs = status_repo.get_pending_with_candidate_name(limit=1)

        if not eligible_jobs:
            # Avoid spamming logs every tick when the queue is empty.
            if is_proctoring_diagnostics_enabled():
                debug_logger.debug(
                    "[SCHEDULER] No eligible jobs — interview_id=N/A | status=IDLE"
                )
            return

        debug_logger.debug(
            f"[SCHEDULER] Eligible jobs found — interview_id=N/A | "
            f"count={len(eligible_jobs)}, dispatching"
        )
        _dispatch_jobs(eligible_jobs)

    except Exception as exc:
        debug_logger.error(
            f"[SCHEDULER] Tick exception — interview_id=N/A{_sched_error_kv(exc)}",
            exc_info=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public start / stop API
# ─────────────────────────────────────────────────────────────────────────────

def start_scheduler() -> None:
    """
    Start the APScheduler background scheduler with an IntervalTrigger
    firing every SCHEDULER_CONFIG.INTERVAL_SECONDS (default 3600 = 1 hour).

    The first tick runs immediately after startup; subsequent ticks are spaced
    by INTERVAL_SECONDS. Safe to call multiple times — returns early if already running.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        debug_logger.debug(
            "[SCHEDULER] Already running — "
            "interview_id=N/A | start_scheduler() ignored"
        )
        return

    try:
        new_scheduler = BackgroundScheduler(
            job_defaults={
                "coalesce": SCHEDULER_CONFIG.JOB_COALESCE,
                "max_instances": SCHEDULER_CONFIG.JOB_MAX_INSTANCES,
            },
            timezone=SCHEDULER_CONFIG.TIMEZONE,
        )
        new_scheduler.add_job(
            func=_scheduler_tick,
            trigger=IntervalTrigger(seconds=SCHEDULER_CONFIG.INTERVAL_SECONDS),
            id=JOB_ID,
            name="Proctoring video processing scheduler",
            replace_existing=True,
            next_run_time=_now_in_scheduler_tz(),
        )
        new_scheduler.start()
        _scheduler = new_scheduler
        _sec = SCHEDULER_CONFIG.INTERVAL_SECONDS
        debug_logger.info(
            f"[SCHEDULER] Started — interview_id=N/A | "
            f"first_run=immediate | interval={_sec}s | MAX_RETRIES={MAX_RETRIES} | "
            f"timezone={SCHEDULER_CONFIG.TIMEZONE}"
        )
    except Exception as exc:
        _scheduler = None
        debug_logger.error(
            f"[SCHEDULER] Failed to start — "
            f"interview_id=N/A{_sched_error_kv(exc)}",
            exc_info=True,
        )
        raise


def stop_scheduler() -> None:
    """Gracefully stop the background scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        debug_logger.debug(
            "[SCHEDULER] Stopped — interview_id=N/A"
        )
    _scheduler = None


def get_scheduler_status() -> dict:
    """Return a status dict (used by the health endpoint)."""
    global _scheduler
    if _scheduler is None or not _scheduler.running:
        return {"running": False, "next_run": None}

    job = _scheduler.get_job(JOB_ID)
    return {
        "running": True,
        "next_run": str(job.next_run_time) if job else None,
        "interval_seconds": SCHEDULER_CONFIG.INTERVAL_SECONDS,
        "timezone": SCHEDULER_CONFIG.TIMEZONE,
        "max_retries": MAX_RETRIES,
    }