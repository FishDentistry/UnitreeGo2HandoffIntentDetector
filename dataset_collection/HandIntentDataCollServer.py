# Run with:
#   python -m uvicorn HandIntentDataCollServer:app --host 0.0.0.0 --port 8000 --workers 1
#
# Optional non-interactive participant ID:
#   PARTICIPANT_ID=P01 python -m uvicorn HandIntentDataCollServer:app --host 0.0.0.0 --port 8000 --workers 1

#HOTKEY MAPPING:
# | Key |       Label | Condition                        |
# | --- | ----------: | -------------------------------- |
# | 1   |     handoff | facing camera, right hand        |
# | 2   |     handoff | facing camera, left hand         |
# | 3   |     handoff | facing away/angled, right hand   |
# | 4   |     handoff | facing away/angled, left hand    |
# | 5   | not_handoff | holding near torso, right hand   |
# | 6   | not_handoff | holding near torso, left hand    |
# | 7   | not_handoff | displaying to camera, right hand |
# | 8   | not_handoff | displaying to camera, left hand  |


import csv
import datetime as dt
import json
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from pynput import keyboard


DATASET_ROOT = Path("handoff_dataset")

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720

DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720

FPS = 6

FRAME_BUFFER_SIZE = 120
JOINT_BUFFER_SIZE = 240

MAX_TIME_SKEW_MS = 200.0
QUEST_JOINTS_LOG_EVERY = 10

# -----------------------------------------------------------------------------
# Granular labels / hotkeys
# -----------------------------------------------------------------------------

SAMPLE_TYPES = {
    "facing_camera_offering_right": {
        "label": 1,
        "label_name": "handoff",
        "condition": "facing_camera_offering",
        "hand": "right",
    },
    "facing_camera_offering_left": {
        "label": 1,
        "label_name": "handoff",
        "condition": "facing_camera_offering",
        "hand": "left",
    },
    "facing_away_offering_right": {
        "label": 1,
        "label_name": "handoff",
        "condition": "facing_away_offering",
        "hand": "right",
    },
    "facing_away_offering_left": {
        "label": 1,
        "label_name": "handoff",
        "condition": "facing_away_offering",
        "hand": "left",
    },
    "holding_torso_not_offering_right": {
        "label": 0,
        "label_name": "not_handoff",
        "condition": "holding_torso_not_offering",
        "hand": "right",
    },
    "holding_torso_not_offering_left": {
        "label": 0,
        "label_name": "not_handoff",
        "condition": "holding_torso_not_offering",
        "hand": "left",
    },
    "displaying_camera_not_offering_right": {
        "label": 0,
        "label_name": "not_handoff",
        "condition": "displaying_camera_not_offering",
        "hand": "right",
    },
    "displaying_camera_not_offering_left": {
        "label": 0,
        "label_name": "not_handoff",
        "condition": "displaying_camera_not_offering",
        "hand": "left",
    },
}

KEY_TO_SAMPLE_TYPE = {
    "1": "facing_camera_offering_right",
    "2": "facing_camera_offering_left",
    "3": "facing_away_offering_right",
    "4": "facing_away_offering_left",
    "5": "holding_torso_not_offering_right",
    "6": "holding_torso_not_offering_left",
    "7": "displaying_camera_not_offering_right",
    "8": "displaying_camera_not_offering_left",
}


def monotonic_ns() -> int:
    return time.monotonic_ns()


def unix_ns() -> int:
    return time.time_ns()


def utc_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def intrinsics_to_dict(intr) -> Dict[str, Any]:
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def sanitize_participant_id(raw: str) -> str:
    """Make the participant ID safe to use in folder paths."""
    cleaned = raw.strip()
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "participant_unset"


def prompt_for_participant_id() -> str:
    """
    Ask for participant ID before capture starts.

    You can skip the prompt by setting PARTICIPANT_ID, e.g.:
      PARTICIPANT_ID=P01 python -m uvicorn HandIntentDataCollServer:app ...
    """
    env_pid = os.environ.get("PARTICIPANT_ID")
    if env_pid:
        participant_id = sanitize_participant_id(env_pid)
        print(f"[collector] Using PARTICIPANT_ID from environment: {participant_id}")
        return participant_id

    try:
        raw = input("Enter participant ID before startup, e.g. P01: ")
    except EOFError:
        raw = "participant_unset"
        print("[collector:warning] Could not read participant ID from stdin; using participant_unset")

    participant_id = sanitize_participant_id(raw)
    print(f"[collector] Participant ID = {participant_id}")
    return participant_id


@dataclass
class RGBDFrame:
    host_monotonic_ns: int
    host_unix_ns: int
    realsense_timestamp_ms: float
    frame_number: int
    color_bgr: np.ndarray
    depth_z16: np.ndarray
    color_intrinsics: Dict[str, Any]
    depth_scale_m_per_unit: float


@dataclass
class JointPacket:
    host_monotonic_ns: int
    host_unix_ns: int
    payload: Dict[str, Any]


class QuestJointsPacket(BaseModel):
    """
    Expected packet from Quest 3.

    This accepts the expanded Unity packet sent by SendQuestJointsToServer,
    including hmd_center_eye, torso, scapula, wrist-twist, and palm joints.

    Example shape:
    {
      "quest_time_ns": 123456789,
      "unity_time": 12.34,
      "frame_id": 44,
      "coordinate_frame": "unity_world",
      "joints": {
        "hmd_center_eye": {...},
        "root": {...},
        "hips": {...},
        "spine_lower": {...},
        "spine_middle": {...},
        "spine_upper": {...},
        "chest": {...},
        "neck": {...},
        "head": {...},
        "left_shoulder": {...},
        "right_shoulder": {...},
        "left_scapula": {...},
        "right_scapula": {...},
        "left_upper_arm": {...},
        "right_upper_arm": {...},
        "left_forearm": {...},
        "right_forearm": {...},
        "left_hand": {...},
        "right_hand": {...},
        "left_wrist_twist": {...},
        "right_wrist_twist": {...},
        "left_palm": {...},
        "right_palm": {...}
      }
    }

    Each joint entry is kept as raw JSON, e.g.:
    {
      "joint_name": "head",
      "bone_id": "Body_Head",
      "valid": true,
      "position": [x, y, z],
      "rotation": [x, y, z, w],
      "euler_angles": [x, y, z],
      "unity_time": 12.34,
      "unity_frame": 44
    }
    """

    quest_time_ns: Optional[int] = None
    unity_time: Optional[float] = None
    frame_id: Optional[int] = None
    coordinate_frame: Optional[str] = "unity_world"
    joints: Dict[str, Any]


class HandoffCollector:
    def __init__(self):
        self.lock = threading.Lock()

        self.frames: deque[RGBDFrame] = deque(maxlen=FRAME_BUFFER_SIZE)
        self.joints: deque[JointPacket] = deque(maxlen=JOINT_BUFFER_SIZE)

        self.latest_frame: Optional[RGBDFrame] = None
        self.latest_joints: Optional[JointPacket] = None
        self.quest_joints_packet_count = 0

        self.stop_event = threading.Event()

        self.realsense_thread: Optional[threading.Thread] = None
        self.save_thread: Optional[threading.Thread] = None
        self.keyboard_listener: Optional[keyboard.Listener] = None

        self.save_requests: queue.Queue[str] = queue.Queue()
        self.last_key_time_ns = 0

        self.dataset_root = DATASET_ROOT
        self.samples_dir = self.dataset_root / "samples"
        # Use a new CSV name so it does not conflict with your old binary-only schema.
        self.index_csv = self.dataset_root / "samples_granular.csv"

        self.participant_id: Optional[str] = None

    def start(self):
        # Prompt before starting RealSense, save worker, and keyboard listener.
        self.participant_id = prompt_for_participant_id()

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create the granular folder tree for the current participant.
        for sample_type, spec in SAMPLE_TYPES.items():
            _ = sample_type  # keeps linters quiet if enabled
            (
                self.samples_dir
                / self.participant_id
                / spec["label_name"]
                / spec["condition"]
                / spec["hand"]
            ).mkdir(parents=True, exist_ok=True)

        self._ensure_csv_header()

        self.stop_event.clear()

        self.realsense_thread = threading.Thread(
            target=self._realsense_loop,
            name="realsense_capture",
            daemon=True,
        )
        self.realsense_thread.start()

        self.save_thread = threading.Thread(
            target=self._save_loop,
            name="save_worker",
            daemon=True,
        )
        self.save_thread.start()

        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()

        print("[collector] Started")
        print(f"[collector] Participant ID: {self.participant_id}")
        print("[collector] Hotkeys:")
        print("  1 = handoff, facing camera, right hand")
        print("  2 = handoff, facing camera, left hand")
        print("  3 = handoff, facing away/angled, right hand")
        print("  4 = handoff, facing away/angled, left hand")
        print("  5 = not_handoff, holding near torso, right hand")
        print("  6 = not_handoff, holding near torso, left hand")
        print("  7 = not_handoff, displaying to camera, right hand")
        print("  8 = not_handoff, displaying to camera, left hand")

    def stop(self):
        self.stop_event.set()

        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()

        print("[collector] Stopping")

    def _ensure_csv_header(self):
        if self.index_csv.exists():
            return

        with self.index_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "participant_id",
                    "label",
                    "label_name",
                    "sample_type",
                    "condition",
                    "hand",
                    "sample_dir",
                    "saved_unix_ns",
                    "frame_unix_ns",
                    "joints_unix_ns",
                    "time_skew_ms",
                    "rgb_path",
                    "depth_path",
                    "joints_path",
                    "meta_path",
                ],
            )
            writer.writeheader()

    def _realsense_loop(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(
            rs.stream.depth,
            DEPTH_WIDTH,
            DEPTH_HEIGHT,
            rs.format.z16,
            FPS,
        )

        config.enable_stream(
            rs.stream.color,
            COLOR_WIDTH,
            COLOR_HEIGHT,
            rs.format.bgr8,
            FPS,
        )

        print("[realsense] Attempting to start pipeline...")
        print(f"[realsense] Requested depth: {DEPTH_WIDTH}x{DEPTH_HEIGHT} z16 @ {FPS}")
        print(f"[realsense] Requested color: {COLOR_WIDTH}x{COLOR_HEIGHT} bgr8 @ {FPS}")

        try:
            profile = pipeline.start(config)
        except Exception as e:
            print("[realsense:error] pipeline.start failed:")
            print(e)
            return

        print("[realsense] Pipeline started successfully.")

        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intrinsics = intrinsics_to_dict(color_stream.get_intrinsics())

            align = rs.align(rs.stream.color)

            print(f"[realsense] depth_scale_m_per_unit = {depth_scale}")

            frame_count = 0
            timeout_count = 0
            missing_count = 0

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError as e:
                    timeout_count += 1
                    print(f"[realsense:timeout] #{timeout_count}: {e}")
                    continue

                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    missing_count += 1
                    print(
                        f"[realsense:missing] #{missing_count}: "
                        f"depth={bool(depth_frame)}, color={bool(color_frame)}"
                    )
                    continue

                host_mono = monotonic_ns()
                host_unix = unix_ns()

                color_bgr = np.asanyarray(color_frame.get_data()).copy()
                depth_z16 = np.asanyarray(depth_frame.get_data()).copy()

                sample = RGBDFrame(
                    host_monotonic_ns=host_mono,
                    host_unix_ns=host_unix,
                    realsense_timestamp_ms=float(color_frame.get_timestamp()),
                    frame_number=int(color_frame.get_frame_number()),
                    color_bgr=color_bgr,
                    depth_z16=depth_z16,
                    color_intrinsics=color_intrinsics,
                    depth_scale_m_per_unit=depth_scale,
                )

                with self.lock:
                    self.frames.append(sample)
                    self.latest_frame = sample

                frame_count += 1

                if frame_count == 1:
                    print("[realsense] First RGB-D frame captured.")
                    print(f"[realsense] color shape: {color_bgr.shape}")
                    print(f"[realsense] depth shape: {depth_z16.shape}")

                elif frame_count % 30 == 0:
                    print(f"[realsense] Captured {frame_count} RGB-D frames.")

        except Exception as e:
            print("[realsense:error] Capture loop crashed:")
            print(e)

        finally:
            pipeline.stop()
            print("[realsense] Pipeline stopped.")

    def add_joints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        packet = JointPacket(
            host_monotonic_ns=monotonic_ns(),
            host_unix_ns=unix_ns(),
            payload=payload,
        )

        with self.lock:
            self.joints.append(packet)
            self.latest_joints = packet
            self.quest_joints_packet_count += 1
            n = len(self.joints)
            total_n = self.quest_joints_packet_count

        if total_n == 1 or total_n % QUEST_JOINTS_LOG_EVERY == 0:
            print(
                f"[quest_joints] Received {total_n} packets "
                f"(buffer_size={n}, frame_id={payload.get('frame_id')})"
            )

        return {
            "ok": True,
            "server_received_monotonic_ns": packet.host_monotonic_ns,
            "server_received_unix_ns": packet.host_unix_ns,
            "joints_buffer_size": n,
        }

    def _on_key_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        now = monotonic_ns()

        # Debounce so holding the key does not save many duplicate samples.
        if now - self.last_key_time_ns < 250_000_000:
            return

        if char in KEY_TO_SAMPLE_TYPE:
            self.last_key_time_ns = now
            sample_type = KEY_TO_SAMPLE_TYPE[char]
            self.save_requests.put(sample_type)

            spec = SAMPLE_TYPES[sample_type]
            print(
                f"[hotkey] {char} pressed -> queued {sample_type} "
                f"(label={spec['label']}, hand={spec['hand']})"
            )

    def _save_loop(self):
        while not self.stop_event.is_set():
            try:
                sample_type = self.save_requests.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                result = self.save_sample(sample_type)
                print(f"[save] {result}")
            except Exception as e:
                print(f"[save:error] {e}")

    def _nearest_joints_to_frame(
        self,
        frame_time_ns: int,
        joint_packets: list[JointPacket],
    ) -> Optional[JointPacket]:
        if not joint_packets:
            return None

        return min(
            joint_packets,
            key=lambda jp: abs(jp.host_monotonic_ns - frame_time_ns),
        )

    def save_sample(self, sample_type: str) -> Dict[str, Any]:
        if sample_type not in SAMPLE_TYPES:
            raise ValueError(
                f"sample_type must be one of {list(SAMPLE_TYPES.keys())}"
            )

        spec = SAMPLE_TYPES[sample_type]
        label = spec["label"]
        label_name = spec["label_name"]
        condition = spec["condition"]
        hand = spec["hand"]

        participant_id = self.participant_id or "participant_unset"

        with self.lock:
            frame = self.latest_frame
            joint_packets = list(self.joints)

        if frame is None:
            raise RuntimeError("No RealSense frame has been captured yet.")

        nearest_joints = self._nearest_joints_to_frame(
            frame.host_monotonic_ns,
            joint_packets,
        )

        if nearest_joints is None:
            raise RuntimeError("No Quest joint packet has been received yet.")

        skew_ms = abs(nearest_joints.host_monotonic_ns - frame.host_monotonic_ns) / 1e6

        if skew_ms > MAX_TIME_SKEW_MS:
            raise RuntimeError(
                f"*** SAMPLE NOT SAVED: TIME SKEW TOO HIGH *** "
                f"The nearest RGB-D frame and Quest joints are {skew_ms:.1f} ms apart, "
                f"which is above the allowed threshold of {MAX_TIME_SKEW_MS:.1f} ms. "
                f"This hotkey press was rejected and no sample files or CSV row were written. "
                f"Check that the Quest joint stream is actively updating and temporally close "
                f"to the RealSense frame before collecting this sample."
            )

        sample_id = utc_id()

        # Granular folder structure:
        # samples/<participant_id>/<handoff|not_handoff>/<condition>/<left|right>/<sample_id>/
        sample_dir = (
            self.samples_dir
            / participant_id
            / label_name
            / condition
            / hand
            / sample_id
        )
        sample_dir.mkdir(parents=True, exist_ok=False)

        rgb_path = sample_dir / "rgb.png"
        depth_path = sample_dir / "depth_z16.png"
        joints_path = sample_dir / "joints_raw.json"
        meta_path = sample_dir / "meta.json"

        # RGB frame is stored as a normal PNG.
        cv2.imwrite(str(rgb_path), frame.color_bgr)

        # Depth is stored as a 16-bit PNG in raw RealSense z16 units.
        cv2.imwrite(str(depth_path), frame.depth_z16)

        with joints_path.open("w") as f:
            json.dump(nearest_joints.payload, f, indent=2)

        saved_unix = unix_ns()

        meta = {
            "sample_id": sample_id,
            "participant_id": participant_id,
            "label": label,
            "label_name": label_name,
            "sample_type": sample_type,
            "condition": condition,
            "hand": hand,
            "saved_unix_ns": saved_unix,
            "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "time_alignment": {
                "frame_host_monotonic_ns": frame.host_monotonic_ns,
                "frame_host_unix_ns": frame.host_unix_ns,
                "joints_host_monotonic_ns": nearest_joints.host_monotonic_ns,
                "joints_host_unix_ns": nearest_joints.host_unix_ns,
                "time_skew_ms": skew_ms,
                "max_allowed_skew_ms": MAX_TIME_SKEW_MS,
                "alignment_basis": "nearest server-side timestamps",
            },
            "realsense": {
                "frame_number": frame.frame_number,
                "realsense_timestamp_ms": frame.realsense_timestamp_ms,
                "color_intrinsics": frame.color_intrinsics,
                "depth_scale_m_per_unit": frame.depth_scale_m_per_unit,
                "depth_encoding": "uint16_z16_raw",
                "depth_to_meters_formula": "depth_m = depth_z16 * depth_scale_m_per_unit",
                "depth_aligned_to": "color",
            },
            "quest_joints": {
                "coordinate_frame": nearest_joints.payload.get("coordinate_frame"),
                "quest_time_ns": nearest_joints.payload.get("quest_time_ns"),
                "frame_id": nearest_joints.payload.get("frame_id"),
                "note": (
                    "Quest joints are temporally paired with the RGB-D frame. "
                    "They are not spatially calibrated or transformed into the "
                    "RealSense camera coordinate frame."
                ),
            },
            "files": {
                "rgb": str(rgb_path),
                "depth_z16": str(depth_path),
                "joints_raw": str(joints_path),
                "meta": str(meta_path),
            },
        }

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        row = {
            "sample_id": sample_id,
            "participant_id": participant_id,
            "label": label,
            "label_name": label_name,
            "sample_type": sample_type,
            "condition": condition,
            "hand": hand,
            "sample_dir": str(sample_dir),
            "saved_unix_ns": saved_unix,
            "frame_unix_ns": frame.host_unix_ns,
            "joints_unix_ns": nearest_joints.host_unix_ns,
            "time_skew_ms": f"{skew_ms:.3f}",
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "joints_path": str(joints_path),
            "meta_path": str(meta_path),
        }

        with self.index_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        return {
            "ok": True,
            "sample_id": sample_id,
            "participant_id": participant_id,
            "label": label,
            "label_name": label_name,
            "sample_type": sample_type,
            "condition": condition,
            "hand": hand,
            "time_skew_ms": round(skew_ms, 3),
            "sample_dir": str(sample_dir),
        }

    def status(self) -> Dict[str, Any]:
        now = monotonic_ns()

        with self.lock:
            frame = self.latest_frame
            joints = self.latest_joints
            n_frames = len(self.frames)
            n_joints = len(self.joints)

        participant_id = self.participant_id or "participant_unset"

        return {
            "has_frame": frame is not None,
            "has_joints": joints is not None,
            "frame_age_ms": (
                None if frame is None else round((now - frame.host_monotonic_ns) / 1e6, 1)
            ),
            "joints_age_ms": (
                None if joints is None else round((now - joints.host_monotonic_ns) / 1e6, 1)
            ),
            "frame_buffer_size": n_frames,
            "joints_buffer_size": n_joints,
            "max_time_skew_ms": MAX_TIME_SKEW_MS,
            "participant_id": participant_id,
            "dataset_root": str(self.dataset_root),
            "samples_dir": str(self.samples_dir / participant_id),
            "index_csv": str(self.index_csv),
            "hotkeys": KEY_TO_SAMPLE_TYPE,
        }


collector = HandoffCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    collector.start()
    yield
    collector.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {
        "message": "Granular handoff intent data collection server",
        "participant_id": collector.participant_id,
        "hotkeys": {
            "1": "handoff: facing camera, right hand",
            "2": "handoff: facing camera, left hand",
            "3": "handoff: facing away/angled, right hand",
            "4": "handoff: facing away/angled, left hand",
            "5": "not_handoff: holding near torso, right hand",
            "6": "not_handoff: holding near torso, left hand",
            "7": "not_handoff: displaying to camera, right hand",
            "8": "not_handoff: displaying to camera, left hand",
        },
        "sample_types": SAMPLE_TYPES,
        "endpoints": [
            "POST /quest_joints",
            "WS /ws/quest_joints",
            "POST /save/{sample_type}",
            "GET /status",
            "GET /server_time",
        ],
    }


@app.get("/status")
def status():
    return collector.status()


@app.get("/server_time")
def server_time():
    return {
        "server_monotonic_ns": monotonic_ns(),
        "server_unix_ns": unix_ns(),
    }


@app.post("/quest_joints")
def receive_quest_joints(packet: QuestJointsPacket):
    if hasattr(packet, "model_dump"):
        payload = packet.model_dump()
    else:
        payload = packet.dict()

    return collector.add_joints(payload)


@app.websocket("/ws/quest_joints")
async def websocket_quest_joints(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()
            result = collector.add_joints(payload)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("[websocket] Quest joint stream disconnected")


@app.post("/save/{sample_type}")
def save_granular_sample(sample_type: str):
    return collector.save_sample(sample_type)
