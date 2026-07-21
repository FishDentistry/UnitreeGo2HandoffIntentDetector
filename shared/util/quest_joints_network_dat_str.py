from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any, Dict, Optional



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