
#Run from repo root with: python -m servers.check_current_handoff_pose_server --quest-mlp-est-path /path/to/mlp_weights_file/file.pth
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from typing import Any, Optional
from pathlib import Path
import threading
import time
import numpy as np
import cv2

from quest_hand_intent_model_est.src.quest_hand_int_inf_wrapper import (
    QuestHandIntentEstInference,
)
from quest_hand_intent_model_est.src.quest_joint_features import (
    extract_quest_joint_features_json, feature_vector_to_joint_poses
)
from quest_hand_intent_model_est.src.find_minimal_perturbation import find_minimal_perturbation
from quest_hand_intent_model_est.src.text_guidance_from_pert import generate_text_guidance_from_perturbation
from quest_hand_intent_model_est.src.posefix_guidance import (
    configure_posefix_for_hand_guidance,
    generate_posefix_guidance,
)
from shared.util.quest_joints_network_dat_str import QuestJointsPacket

from model_training_and_implementation.src.dino_detector import DINOObjectDetector
from robot_fov_estimation.src.go2_yolo_det_wrapper import YOLOGo2Detector
from robot_fov_estimation.src.robot_detector_tracker import (
    RobotDetectorTracker,
    OSTrackAdapter,
    RobotTrackResult,
    draw_result,
)


# Replace this with the specific prompt you already selected.
ROBOT_DETECTOR_CLASS_NAMES = [
    "robot dog"
]




def format_perturbation_response(
    original_joints: np.ndarray,
    perturbed_joints: np.ndarray,
    perturbations_needed: bool = True,
    convert_to_world_positions: bool = True,
    shoulder_midpoint: list[float] = [0.0, 0.0, 0.0],
    generate_text_guidance: bool = True
) -> dict[str, Any]:
    """
    Convert the output of find_minimal_perturbation() into the response
    format expected by Unity.

    Positions are final joint positions relative to the shoulder midpoint.

    When rotations are present in the feature vector, they are included in
    the response. When the model uses position-only features, the rotation
    dictionary remains empty.
    """
    perturbed_joint_poses_dict = feature_vector_to_joint_poses(
        perturbed_joints
    )

    joint_position_perturbations: dict[
        str,
        list[float],
    ] = {}

    joint_rotation_perturbations: dict[
        str,
        list[float],
    ] = {}


    for joint_name, joint_pose in perturbed_joint_poses_dict.items():
        position = joint_pose["position"]

        joint_position_perturbations[joint_name] = (
            position.astype(float).tolist()
        )

        rotation = joint_pose["rotation"]

        if rotation is not None:
            joint_rotation_perturbations[joint_name] = (
                rotation.astype(float).tolist()
            )
        if(convert_to_world_positions):
            joint_position_perturbations[joint_name][0] += shoulder_midpoint[0]
            joint_position_perturbations[joint_name][1] += shoulder_midpoint[1]
            joint_position_perturbations[joint_name][2] += shoulder_midpoint[2]

    print(joint_position_perturbations)

    text_guidance = ""
    if generate_text_guidance:
        original_joint_poses_dict = feature_vector_to_joint_poses(
                original_joints
            )
        text_guidance = generate_posefix_guidance(
                original_joints=(
                    original_joint_poses_dict
                ),
                perturbed_joints=(
                    perturbed_joint_poses_dict
                ),
                simplified_instructions=True,
            )
        
    return {
        "perturbations_needed": perturbations_needed,
        "joint_position_perturbations":
            joint_position_perturbations,
        "joint_rotation_perturbations":
            joint_rotation_perturbations,
        "text_guidance": text_guidance,
    }

def format_robot_tracking_result(
    result: Optional[RobotTrackResult],
) -> Optional[dict[str, Any]]:
    """Convert RobotTrackResult into a JSON-serializable dictionary."""
    if result is None:
        return None

    return {
        "found": result.found,
        "state": result.state.value,
        "bbox_xyxy": (
            list(result.bbox_xyxy)
            if result.bbox_xyxy is not None
            else None
        ),
        "frame_id": result.frame_id,
        "frame_timestamp": result.frame_timestamp,
        "completed_timestamp": result.completed_timestamp,
        "processing_time_ms": result.processing_time_ms,
        "end_to_end_latency_ms": result.end_to_end_latency_ms,
        "source": result.source,
        "detector_score": result.detector_score,
        "detector_label": result.detector_label,
        "tracker_verified": result.tracker_verified,
        "acquisition_candidate_count":
            result.acquisition_candidate_count,
        "replacement_candidate_count":
            result.replacement_candidate_count,
        "verification_misses": result.verification_misses,
    }


def create_app(robot_obj_det_weights_path:str, model_path: str) -> FastAPI:
    app = FastAPI()

    model = QuestHandIntentEstInference(
        checkpoint_path=model_path,
    )

    configure_posefix_for_hand_guidance()

    # ------------------------------------------------------------------
    # Robot detector/tracker initialization
    # ------------------------------------------------------------------
    #
    # These objects are created once and reused by every /robot_frame
    # request. Do not create DINO or CSRT separately inside the endpoint.
    #
    dino_detector = DINOObjectDetector(confidence=0.05)

    if(robot_obj_det_weights_path is not None):
        yolo_detector = YOLOGo2Detector(
            confidence=0.05,
            iou_threshold=0.50,
            image_size=640,
            device=0,
            weights_path=robot_obj_det_weights_path
        )
    else:
        yolo_detector = YOLOGo2Detector(
            confidence=0.05,
            iou_threshold=0.50,
            image_size=960,
            device=0,
        )

    ostrack = OSTrackAdapter()
    robot_tracker = RobotDetectorTracker(
        detector=yolo_detector,
        ostrack = ostrack,
        class_names=ROBOT_DETECTOR_CLASS_NAMES,
        verification_interval_seconds=0.15,
        verification_iou_threshold=0.40,
        acquisition_confirmations=2,
        acquisition_iou_threshold=0.20,
        max_verification_misses=10,
        replacement_confirmations=2,
        replacement_iou_threshold=0.20,
        min_detection_score=0.50,
    )

    # RobotDetectorTracker and OpenCV CSRT are stateful. Only one request may
    # update the tracker at a time.
    app.state.robot_tracker = robot_tracker
    app.state.robot_tracking_lock = threading.Lock()
    app.state.latest_robot_tracking_result = None
    app.state.last_robot_frame_id = -1

    def process_robot_frame(
        image_bytes: bytes,
        frame_id: int,
        capture_timestamp_unix: Optional[float],
    ) -> dict[str, Any]:
        """
        Decode and process one image.

        This function runs in FastAPI/Starlette's thread pool so DINO and
        OpenCV do not block the asyncio event loop.
        """
        tracking_lock: threading.Lock = (
            app.state.robot_tracking_lock
        )

        # Do not let requests form an old-frame backlog. If one frame is
        # already being processed, this request immediately receives the
        # latest completed result.
        if not tracking_lock.acquire(blocking=False):
            return {
                "accepted": False,
                "reason": "tracker_busy",
                "submitted_frame_id": frame_id,
                "result": format_robot_tracking_result(
                    app.state.latest_robot_tracking_result
                ),
            }

        try:
            # frame_id must increase for a given Quest stream.
            if frame_id <= app.state.last_robot_frame_id:
                return {
                    "accepted": False,
                    "reason": "stale_or_out_of_order_frame",
                    "submitted_frame_id": frame_id,
                    "last_processed_frame_id":
                        app.state.last_robot_frame_id,
                    "result": format_robot_tracking_result(
                        app.state.latest_robot_tracking_result
                    ),
                }

            encoded_image = np.frombuffer(
                image_bytes,
                dtype=np.uint8,
            )

            frame_bgr = cv2.imdecode(
                encoded_image,
                cv2.IMREAD_COLOR,
            )

            if frame_bgr is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Request body could not be decoded as "
                        "a JPEG or PNG image."
                    ),
                )

            # Use a client timestamp only when the Quest and server clocks
            # are synchronized. Otherwise, omit it and server receive time
            # is used.
            frame_timestamp = (
                float(capture_timestamp_unix)
                if capture_timestamp_unix is not None
                else time.time()
            )

            result = app.state.robot_tracker.process_frame(
                frame_bgr=frame_bgr,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
            )
            # result = app.state.robot_tracker.process_frame_detector_only(
            #         frame_bgr=frame_bgr,
            #         frame_id=frame_id,
            #         frame_timestamp=frame_timestamp,
            #     )

            annotated_frame = draw_result(frame_bgr, result)

            output_path = (
                Path(__file__).resolve().parent
                / "latest_robot_detection.jpg"
            )

            saved = cv2.imwrite(
                str(output_path),
                annotated_frame,
            )

            print(
                f"Annotated frame save: success={saved}, "
                f"exists={output_path.exists()}, "
                f"path={output_path}",
                flush=True,
            )



            app.state.last_robot_frame_id = frame_id
            app.state.latest_robot_tracking_result = result

            res = {
                "accepted": True,
                "reason": "processed",
                "submitted_frame_id": frame_id,
                "image_width": int(frame_bgr.shape[1]),
                "image_height": int(frame_bgr.shape[0]),
                "result": format_robot_tracking_result(result),
            }
            print(res)

            return res

        finally:
            tracking_lock.release()

    configure_posefix_for_hand_guidance()


    @app.post("/quest_joints")
    def receive_quest_joints(packet: QuestJointsPacket):
        print ("Received packet")
        if hasattr(packet, "model_dump"):
            payload = packet.model_dump()
        else:
            payload = packet.dict()

        # Shoulder positions from the same packet used for inference.
        left_shoulder = payload["joints"]["left_shoulder"]["position"]
        right_shoulder = payload["joints"]["right_shoulder"]["position"]

        shoulder_midpoint = [
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
            (left_shoulder[2] + right_shoulder[2]) / 2.0,
        ]

        joint_feat_vec = extract_quest_joint_features_json(
            payload,
            include_rotations=model.include_rotations,
        )

        original_prediction = model.predict_features(
            joint_feat_vec
        )

        target_intent = True

        if original_prediction.is_handoff == target_intent:
            return {
                "perturbations_needed": False,
                "joint_position_perturbations": {},
                "joint_rotation_perturbations": {},
                "text_guidance": "",
            }

        perturbed_features = find_minimal_perturbation(
            model=model,
            joint_feat_vec=joint_feat_vec,
            target_intent=True,
            classification_weight=1000.0,
            reachability_weight=1000.0,
            probability_margin=0.05,
            max_iterations=500,
        )
        

        response = format_perturbation_response(
            original_joints=joint_feat_vec,
            perturbed_joints=perturbed_features,
            perturbations_needed=True,
            shoulder_midpoint=shoulder_midpoint
        )

        # Convert shoulder-relative positions back into Unity world positions relative to the initial shoulder midpoint.
        # for position in response[
        #     "joint_position_perturbations"
        # ].values():
        #     position[0] += shoulder_midpoint[0]
        #     position[1] += shoulder_midpoint[1]
        #     position[2] += shoulder_midpoint[2]

        return response

    @app.post("/robot_tracking")
    async def try_track_robot(
        request: Request,
        frame_id: int = Query(
            ...,
            ge=0,
            description=(
                "Strictly increasing frame number for the Quest stream."
            ),
        ),
        capture_timestamp_unix: Optional[float] = Query(
            default=None,
            description=(
                "Optional Unix timestamp, in seconds, when the frame "
                "was captured."
            ),
        ),
    ):
        """
        Receive one encoded Quest RGB image and return robot tracking data.

        Request body:
            Raw JPEG or PNG bytes.

        Required query parameter:
            frame_id

        Optional query parameter:
            capture_timestamp_unix
        """
        image_bytes = await request.body()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The request body is empty.",
            )

        return await run_in_threadpool(
            process_robot_frame,
            image_bytes,
            frame_id,
            capture_timestamp_unix,
        )

    # Optional endpoint for forcing a clean reacquisition.
    @app.post("/reset_robot_tracker")
    def reset_robot_tracker():
        tracking_lock: threading.Lock = (
            app.state.robot_tracking_lock
        )

        with tracking_lock:
            app.state.robot_tracker.reset()
            app.state.latest_robot_tracking_result = None
            app.state.last_robot_frame_id = -1

        return {
            "success": True,
            "state": "searching",
        }

    return app
        



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quest-mlp-est-path",
        type=str,
        required=True,
        help="Path to the trained handoff-classification model.",
    )

    parser.add_argument(
            "--robot-obj-det-weights-path",
            type=str,
            default=None,
            help="Path to the trained robot object detection model weights if using YOLO or similar.",
        )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    args = parser.parse_args()

    app = create_app(args.robot_obj_det_weights_path, args.quest_mlp_est_path)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()