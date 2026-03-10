import signal
import time

from sqlmodel import Session

from backend.decision.handler import handle_track_closed
from backend.fusion.models import Read
from backend.fusion.pipeline import process_frame
from backend.fusion.tracker import TrackManager
from backend.ingestion.camera import CameraSource
from backend.ingestion.motion import MotionDetector
from backend.utils.csv_logger import append_reads
from backend.utils.logging import get_logger

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
    ):
        self._cameras = cameras
        self._detector = detector
        self._ocr = ocr
        self._track_manager = track_manager
        self._session = session
        self._csv_path = csv_path
        self._min_interval = 1.0 / processing_fps if processing_fps > 0 else 0
        self._running = False

        self._motion_detectors: dict[str, MotionDetector] = {}
        if use_motion:
            for cam in cameras:
                self._motion_detectors[cam.camera_id] = MotionDetector(
                    threshold=motion_threshold, warmup_frames=motion_warmup,
                )

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

    def _run_loop(self) -> None:
        while self._running:
            loop_start = time.monotonic()
            all_reads: list[Read] = []
            now = time.time()
            any_active = False

            for cam in self._cameras:
                if not cam.active:
                    continue
                any_active = True

                frame_data = cam.read()
                if frame_data is None:
                    continue

                frame, timestamp = frame_data

                motion_det = self._motion_detectors.get(cam.camera_id)
                if motion_det and not motion_det.has_motion(frame):
                    continue

                reads = process_frame(
                    frame, self._detector, self._ocr,
                    camera_id=cam.camera_id, timestamp=timestamp,
                )
                all_reads.extend(reads)

            if not any_active:
                log.info("all_cameras_exhausted")
                break

            if all_reads:
                if self._csv_path:
                    append_reads(self._csv_path, all_reads)
                log.info("reads", count=len(all_reads),
                         texts=[f"{r.class_name}={r.text}" for r in all_reads])

            result = self._track_manager.update(all_reads, now=now)
            if result is not None:
                track = self._track_manager.completed[-1][0]
                event = handle_track_closed(
                    track_id=track.id, results=result, session=self._session,
                )
                self._print_event(event)

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
            event = handle_track_closed(
                track_id=track.id, results=result, session=self._session,
            )
            self._print_event(event, final=True)

        log.info("orchestrator_stopped",
                 tracks_completed=len(self._track_manager.completed))

    def _print_event(self, event, final: bool = False) -> None:
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
