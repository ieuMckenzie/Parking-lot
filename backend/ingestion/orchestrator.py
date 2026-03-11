import signal
import time

import cv2
import numpy as np
from sqlmodel import Session

from backend.config import settings
from backend.decision.handler import handle_track_closed
from backend.fusion.models import FusionResult, Read
from backend.fusion.pipeline import Annotation, process_frame
from backend.fusion.tracker import TrackManager
from backend.ingestion.camera import CameraSource
from backend.ingestion.motion import MotionDetector
from backend.utils.csv_logger import append_reads
from backend.utils.logging import get_logger
from backend.utils.visualization import draw_annotations, draw_status

log = get_logger("orchestrator")


class GateOrchestrator:
    """Manages camera capture, motion filtering, and pipeline processing."""

    def __init__(
        self,
        cameras: list[CameraSource],
        detector,
        ocr,
        track_manager: TrackManager,
        session: Session,
        use_motion: bool = True,
        motion_threshold: float = 0.01,
        motion_warmup: int = 30,
        csv_path: str | None = None,
        processing_fps: float = 10.0,
        display: bool = False,
        output_path: str | None = None,
        quiet: bool = False,
        show_banner: bool = True,
        ocr_every_n_frames: int = 1,
    ):
        self._cameras = cameras
        self._detector = detector
        self._ocr = ocr
        self._track_manager = track_manager
        self._session = session
        self._csv_path = csv_path
        self._min_interval = 1.0 / processing_fps if processing_fps > 0 else 0
        self._running = False
        self._display = display
        self._output_path = output_path
        self._video_writer: cv2.VideoWriter | None = None
        self._quiet = quiet
        self._show_banner = show_banner
        self._ocr_every_n_frames = max(1, ocr_every_n_frames)
        self._camera_frame_counts: dict[str, int] = {cam.camera_id: 0 for cam in cameras}
        self._duplicate_cooldown = settings.fusion.duplicate_event_cooldown_seconds
        self._duplicate_key_classes = set(settings.fusion.duplicate_event_key_classes)
        self._recent_event_signatures: dict[tuple[str, ...], float] = {}

        self._motion_detectors: dict[str, MotionDetector] = {}
        if use_motion:
            for cam in cameras:
                self._motion_detectors[cam.camera_id] = MotionDetector(
                    threshold=motion_threshold, warmup_frames=motion_warmup,
                )

        # Display state
        self._last_decision: str | None = None
        self._decision_expire: float = 0.0
        self._fps_smooth: float = 0.0

    def start(self) -> None:
        """Start all cameras and run the processing loop."""
        self._running = True
        self._setup_signals()

        for cam in self._cameras:
            cam.start()
            log.info("camera_started", camera_id=cam.camera_id)

        log.info("orchestrator_started", num_cameras=len(self._cameras))

        try:
            self._run_loop()
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the processing loop to stop."""
        self._running = False

    def _setup_signals(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Can't set signal handlers outside main thread (e.g., in tests)
            pass

    def _signal_handler(self, signum, frame) -> None:
        log.info("shutdown_signal", signal=signum)
        self._running = False

    def _init_video_writer(self, frame: np.ndarray) -> None:
        if self._video_writer is not None or self._output_path is None:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(self._output_path, fourcc, 10.0, (w, h))
        log.info("video_writer_started", path=self._output_path, size=f"{w}x{h}")

    def _run_loop(self) -> None:
        while self._running:
            loop_start = time.monotonic()
            all_reads: list[Read] = []
            all_annotations: list[Annotation] = []
            now = time.time()
            any_active = False
            display_frame: np.ndarray | None = None

            for cam in self._cameras:
                if not cam.active:
                    continue
                any_active = True

                frame_data = cam.read()
                if frame_data is None:
                    continue

                frame, timestamp = frame_data
                self._camera_frame_counts[cam.camera_id] = self._camera_frame_counts.get(cam.camera_id, 0) + 1
                frame_count = self._camera_frame_counts[cam.camera_id]

                # Keep latest frame for display (even if motion-skipped)
                if self._display or self._output_path:
                    display_frame = frame

                motion_det = self._motion_detectors.get(cam.camera_id)
                if motion_det and not motion_det.has_motion(frame):
                    continue

                run_ocr = (frame_count - 1) % self._ocr_every_n_frames == 0

                lock_classes = set(settings.ocr.lock_after_read_classes)
                lock_min_conf = settings.ocr.lock_min_confidence
                skip_ocr_classes: set[str] = set()
                active_track = self._track_manager.active_track
                if active_track is not None:
                    for read in active_track.reads:
                        if read.class_name in lock_classes and read.confidence >= lock_min_conf:
                            skip_ocr_classes.add(read.class_name)

                reads, annotations = process_frame(
                    frame, self._detector, self._ocr,
                    camera_id=cam.camera_id, timestamp=timestamp,
                    skip_ocr_classes=skip_ocr_classes,
                    run_ocr=run_ocr,
                )
                all_reads.extend(reads)
                all_annotations.extend(annotations)

            if not any_active:
                log.info("all_cameras_exhausted")
                break

            if all_reads:
                if self._csv_path:
                    append_reads(self._csv_path, all_reads)
                if not self._quiet:
                    log.info("reads", count=len(all_reads),
                             texts=[f"{r.class_name}={r.text}" for r in all_reads])

            result = self._track_manager.update(all_reads, now=now)
            decision_text = None
            if result is not None:
                track = self._track_manager.completed[-1][0]
                is_duplicate = self._is_duplicate_event(result, now)
                event = handle_track_closed(
                    track_id=track.id, results=result, session=self._session,
                )
                if not is_duplicate:
                    self._print_event(event)
                    decision_text = event.decision.value
                    self._last_decision = decision_text
                    self._decision_expire = time.monotonic() + 3.0

            # Render display / output
            if display_frame is not None and (self._display or self._output_path):
                elapsed = time.monotonic() - loop_start
                current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self._fps_smooth = 0.8 * self._fps_smooth + 0.2 * current_fps

                active_track = self._track_manager.active_track
                track_reads = len(active_track.reads) if active_track else 0

                banner = self._last_decision if self._show_banner and time.monotonic() < self._decision_expire else None

                annotated = draw_annotations(display_frame, all_annotations)
                annotated = draw_status(
                    annotated,
                    decision=banner,
                    fps=self._fps_smooth,
                    track_reads=track_reads,
                )

                if self._display:
                    cv2.imshow("GateVision", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._running = False

                if self._output_path:
                    self._init_video_writer(annotated)
                    if self._video_writer is not None:
                        self._video_writer.write(annotated)

            elapsed = time.monotonic() - loop_start
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

    def _shutdown(self) -> None:
        log.info("shutting_down")
        for cam in self._cameras:
            cam.stop()
            log.info("camera_stopped", camera_id=cam.camera_id)

        result = self._track_manager.flush()
        if result is not None:
            track = self._track_manager.completed[-1][0]
            now = time.time()
            is_duplicate = self._is_duplicate_event(result, now)
            event = handle_track_closed(
                track_id=track.id, results=result, session=self._session,
            )
            if not is_duplicate:
                self._print_event(event, final=True)

        if self._video_writer is not None:
            self._video_writer.release()
            log.info("video_writer_closed", path=self._output_path)

        if self._display:
            cv2.destroyAllWindows()

        log.info("orchestrator_stopped",
                 tracks_completed=len(self._track_manager.completed))

    def _print_event(self, event, final: bool = False) -> None:
        if self._quiet:
            return
        label = "GATE EVENT (final flush)" if final else "GATE EVENT"
        print(f"\n  === {label} ===")
        print(f"  Decision: {event.decision.value}")
        print(f"  Reason:   {event.decision_reason}")
        if event.usdot_number:
            print(f"  USDOT:    {event.usdot_number}")
        if event.license_plate:
            print(f"  Plate:    {event.license_plate}")
        if event.trailer_number:
            print(f"  Trailer:  {event.trailer_number}")
        print()

    def _is_duplicate_event(self, results: list[FusionResult], now: float) -> bool:
        if self._duplicate_cooldown <= 0:
            return False

        # Build signature from configured identity classes.
        parts = sorted(
            f"{r.class_name}:{r.value}"
            for r in results
            if r.class_name in self._duplicate_key_classes and r.value
        )
        if not parts:
            return False

        signature = tuple(parts)

        # Prune expired signatures to keep memory bounded.
        expired = [sig for sig, seen_at in self._recent_event_signatures.items() if (now - seen_at) > self._duplicate_cooldown]
        for sig in expired:
            del self._recent_event_signatures[sig]

        last_seen = self._recent_event_signatures.get(signature)
        self._recent_event_signatures[signature] = now
        return last_seen is not None and (now - last_seen) <= self._duplicate_cooldown
