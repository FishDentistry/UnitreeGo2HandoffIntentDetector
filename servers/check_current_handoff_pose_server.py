# Run from repo root, e.g.:
# python -m servers.check_current_handoff_pose_server \
#     --quest-model-type tabm \
#     --quest-model-features-type keypoints_projections
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
from quest_hand_intent_model_est.src.quest_hand_int_tabm_inf_wrapper import QuestHandIntentTabMEstInference as TabMModel
from quest_hand_intent_model_est.src.quest_joint_features import (
    extract_quest_joint_features_json, feature_vector_to_joint_poses,
    construct_pose_img_from_joint_feature_vector
)
from quest_hand_intent_model_est.src.find_minimal_perturbation import find_minimal_perturbation
from quest_hand_intent_model_est.src.text_guidance_from_pert import generate_text_guidance_from_perturbation
#Uncomment for text guidance
# from quest_hand_intent_model_est.src.posefix_guidance import (
#     configure_posefix_for_hand_guidance,
#     generate_posefix_guidance,
# )
from shared.util.quest_joints_network_dat_str import QuestJointsPacket

from model_training_and_implementation.src.resnet_encoder import ResNet18ImageEncoder
from robot_fov_estimation.src.orient_anything_forward_estimation_wrapper import (
    OrientAnythingGo2ForwardEstimator,
    OrientationEstimate,
)
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

# Run the expensive visual heading estimator independently of the tracker.
# 0.10 s corresponds to a maximum requested heading-update rate of 10 Hz.
ROBOT_HEADING_INTERVAL_SECONDS = 0.10

QUEST_MODEL_TYPES = ("mlp", "tabm")
QUEST_MODEL_FEATURES_TYPES = (
    "keypoints_projections",
    "keypoints_resnet",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEST_OUTPUT_ROOT = (
    REPO_ROOT
    / "quest_hand_intent_model_est"
    / "outputs"
)


def resolve_quest_estimator_path(
    quest_model_type: str,
    quest_model_features_type: str,
    override_path: Optional[str] = None,
) -> Path:
    """Resolve the Quest student checkpoint path.

    If override_path is supplied, it takes precedence. Otherwise search the
    normal Quest model output directory for a checkpoint whose variation
    folder matches the selected model type and student feature type.
    """
    model_type = str(quest_model_type).strip().lower()
    features_type = str(quest_model_features_type).strip().lower()

    if model_type not in QUEST_MODEL_TYPES:
        raise ValueError(
            f"Invalid quest model type: {quest_model_type}. "
            f"Expected one of {QUEST_MODEL_TYPES}."
        )

    if features_type not in QUEST_MODEL_FEATURES_TYPES:
        raise ValueError(
            f"Invalid quest model features type: {quest_model_features_type}. "
            f"Expected one of {QUEST_MODEL_FEATURES_TYPES}."
        )

    if override_path is not None:
        resolved = Path(override_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Quest estimator checkpoint does not exist: {resolved}"
            )
        return resolved

    if model_type == "mlp":
        model_root = QUEST_OUTPUT_ROOT / "quest_hand_intent_mlp_weights"
        checkpoint_name = "MLP.pth"
    else:
        model_root = QUEST_OUTPUT_ROOT / "quest_hand_intent_tabm"
        checkpoint_name = "TabM.pth"

    variation_pattern = (
        f"model-{model_type}__*"
        f"student_features-{features_type}__*"
    )

    matches = [
        path
        for path in model_root.glob(
            f"{variation_pattern}/{checkpoint_name}"
        )
        if path.is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            "Could not automatically locate a Quest estimator checkpoint. "
            f"Searched under {model_root} for model_type={model_type}, "
            f"student_features_type={features_type}, checkpoint={checkpoint_name}. "
            "Pass --quest-estimator-path to override automatic resolution."
        )

    # There can be multiple matching variation directories if, for example,
    # different teacher features or tuned loss weights were used. Prefer the
    # most recently modified checkpoint; an explicit path can always override
    # this behavior.
    matches.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    selected = matches[0].resolve()

    if len(matches) > 1:
        print(
            "Multiple Quest estimator checkpoints matched "
            f"model_type={model_type}, features_type={features_type}. "
            f"Using the most recently modified checkpoint: {selected}"
        )

    return selected


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
    # if generate_text_guidance:
    #     original_joint_poses_dict = feature_vector_to_joint_poses(
    #             original_joints
    #         )
    #     text_guidance = generate_posefix_guidance(
    #             original_joints=(
    #                 original_joint_poses_dict
    #             ),
    #             perturbed_joints=(
    #                 perturbed_joint_poses_dict
    #             ),
    #             simplified_instructions=True,
    #         )
        
    return {
        "perturbations_needed": perturbations_needed,
        "joint_position_perturbations":
            joint_position_perturbations,
        "joint_rotation_perturbations":
            joint_rotation_perturbations,
        "text_guidance": text_guidance,
    }

def _orientation_angle_sin_cos(
    orientation: Optional[OrientationEstimate],
) -> list[float]:
    """Return [sin(yaw), cos(yaw)] for the selected signed-yaw estimate."""
    if (
        orientation is None
        or not orientation.robot_detected
        or orientation.relative_yaw_deg is None
    ):
        return [0.0, 0.0]

    angle_rad = np.deg2rad(float(orientation.relative_yaw_deg))
    return [
        float(np.sin(angle_rad)),
        float(np.cos(angle_rad)),
    ]


def format_robot_tracking_result(
    result: Optional[RobotTrackResult],
    orientation: Optional[OrientationEstimate] = None,
    orientation_error: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Convert tracking + visual forward estimation into a JSON-safe result.

    `angle_between_cam_and_robot_forward` is the Orient Anything wrapper's
    signed relative yaw:

        SignedAngle(camera_to_robot_planar, robot_forward_planar, +Y)

    The legacy angle fields are preserved for Unity compatibility. Use
    `forward_estimation_valid` before consuming them; [0, 0] is returned for
    `angle_sin_cos` when no valid forward estimate is available.
    """
    if result is None:
        return None

    forward_valid = (
        result.found
        and orientation is not None
        and orientation.robot_detected
        and orientation.relative_yaw_deg is not None
    )

    relative_yaw = (
        float(orientation.relative_yaw_deg)
        if forward_valid
        else 0.0
    )

    orientation_payload = (
        orientation.to_dict()
        if orientation is not None
        else None
    )

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

        # Existing fields, now populated by the fine-tuned OA model.
        "forward_estimation_valid": bool(forward_valid),
        "angle_between_cam_and_robot_forward": relative_yaw,
        "angle_sin_cos": _orientation_angle_sin_cos(orientation),

        # Rich diagnostics / optional Quest-world forward vector.
        "forward_estimation": orientation_payload,
        "forward_estimation_error": orientation_error,
    }


def create_app(
    robot_obj_det_weights_path: Optional[str],
    model_path: str,
    quest_model_type: str,
    quest_model_features_type: str,
    robot_forward_estimator_path: Optional[str] = None,
    robot_forward_decoder: str = "mean",
    user_handedness: Optional[str] = "both",
) -> FastAPI:
    app = FastAPI()

    quest_model_type = str(quest_model_type).strip().lower()
    quest_model_features_type = str(quest_model_features_type).strip().lower()

    if quest_model_type == "mlp":
        model = QuestHandIntentEstInference(
            checkpoint_path=model_path,
        )
    elif quest_model_type == "tabm":
        model = TabMModel(
            checkpoint_path=model_path,
        )
    else:
        raise ValueError(
            f"Invalid quest_model_type: {quest_model_type}"
        )
    
    
    #configure_posefix_for_hand_guidance()

    # ------------------------------------------------------------------
    # Quest-side image models
    # ------------------------------------------------------------------
    resnet_encoder = None
    if quest_model_features_type == "keypoints_resnet":
        resnet_encoder = ResNet18ImageEncoder(
            pretrained=True,
            device="cuda",
            l2_normalize=True,
        )

    # Load the FINAL all-data Orient Anything checkpoint once. The wrapper
    # restores the exact YOLO settings saved by the training script. If an
    # explicit detector checkpoint is supplied to this server, it overrides
    # the checkpoint-recorded YOLO path while retaining the saved confidence,
    # IoU, image size, and padding settings.
    robot_forward_estimator = OrientAnythingGo2ForwardEstimator(
        checkpoint_path=robot_forward_estimator_path,
        decoder=robot_forward_decoder,
        yolo_weights_path=robot_obj_det_weights_path,
    )

    if not robot_forward_estimator.yolo_crop_enabled:
        raise RuntimeError(
            "The deployed forward-estimation checkpoint has YOLO cropping "
            "disabled. This server expects the deployment-matched YOLO crop "
            "pipeline used by the current training script."
        )

    if robot_forward_estimator.detector is None:
        raise RuntimeError(
            "OrientAnythingGo2ForwardEstimator did not create a YOLO detector."
        )

    # IMPORTANT: share the wrapper's YOLO detector with RobotDetectorTracker.
    # This avoids loading a second YOLO model and guarantees that tracking and
    # orientation estimation use the checkpoint's deployment-matched detector.
    yolo_detector = robot_forward_estimator.detector

    print(
        "Robot forward estimator configuration:",
        robot_forward_estimator.checkpoint_summary(),
        flush=True,
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
        min_detection_score=0.6,
    )

    # RobotDetectorTracker / OSTrack is stateful. Only one request may update
    # the tracker at a time. Heading estimation has a separate lock so an
    # expensive Orient Anything inference does not keep the tracker locked.
    app.state.robot_tracker = robot_tracker
    app.state.robot_tracking_lock = threading.Lock()
    app.state.robot_heading_lock = threading.Lock()
    app.state.robot_heading_interval_seconds = ROBOT_HEADING_INTERVAL_SECONDS
    app.state.last_robot_heading_started_monotonic = float("-inf")

    app.state.latest_robot_tracking_result = None
    app.state.latest_robot_orientation_estimate = None
    app.state.latest_robot_orientation_error = None
    app.state.last_robot_frame_id = -1
    app.state.latest_robot_camera_position = None
    app.state.latest_robot_camera_rotation = None
    app.state.latest_robot_camera_intrinsics = None

    def process_robot_frame(
        image_bytes: bytes,
        frame_id: int,
        capture_timestamp_unix: Optional[float],
        camera_position: Optional[dict[str, float]],
        camera_rotation: Optional[dict[str, float]],
        camera_intrinsics: Optional[dict[str, float]],
    ) -> dict[str, Any]:
        """
        Decode and process one image.

        Tracking and heading estimation intentionally run under different locks:

        1. The stateful tracker processes the newest accepted frame and publishes
           its result while holding robot_tracking_lock.
        2. robot_tracking_lock is released immediately after that tracker update.
        3. Orient Anything runs only when its independent interval is due and its
           own non-blocking lock is available.
        4. Frames between heading updates reuse the latest valid heading.

        This lets newer Quest frames continue updating robot position/bbox while
        an expensive heading estimate is being calculated.
        """
        tracking_lock: threading.Lock = app.state.robot_tracking_lock
        heading_lock: threading.Lock = app.state.robot_heading_lock

        # Do not let requests form an old-frame backlog. If the stateful tracker
        # is already processing a frame, return the newest completed result.
        if not tracking_lock.acquire(blocking=False):
            latest_result = app.state.latest_robot_tracking_result
            latest_orientation = app.state.latest_robot_orientation_estimate
            latest_orientation_error = app.state.latest_robot_orientation_error

            return {
                "accepted": False,
                "reason": "tracker_busy",
                "submitted_frame_id": frame_id,
                "result": format_robot_tracking_result(
                    latest_result,
                    latest_orientation,
                    latest_orientation_error,
                ),
            }

        # ------------------------------------------------------------------
        # Fast/stateful tracking section.
        #
        # Keep this lock only as long as needed to decode the accepted frame,
        # update the tracker, and publish the new tracking state.
        # ------------------------------------------------------------------
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
                        app.state.latest_robot_tracking_result,
                        app.state.latest_robot_orientation_estimate,
                        app.state.latest_robot_orientation_error,
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
            # are synchronized. Otherwise, server receive time is used.
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

            # Publish the tracking result BEFORE any heading inference. New
            # requests can use this position/bbox as soon as the lock releases.
            app.state.last_robot_frame_id = frame_id
            app.state.latest_robot_tracking_result = result
            app.state.latest_robot_camera_position = camera_position
            app.state.latest_robot_camera_rotation = camera_rotation
            app.state.latest_robot_camera_intrinsics = camera_intrinsics

        finally:
            tracking_lock.release()

        # ------------------------------------------------------------------
        # Slower visual heading estimation.
        #
        # This section is deliberately outside robot_tracking_lock. It is also
        # throttled and guarded by its own non-blocking lock so there can be at
        # most one Orient Anything inference in flight.
        # ------------------------------------------------------------------
        orientation_estimate: Optional[OrientationEstimate] = (
            app.state.latest_robot_orientation_estimate
        )
        orientation_error: Optional[str] = (
            app.state.latest_robot_orientation_error
        )

        if result.found and result.bbox_xyxy is not None:
            now_monotonic = time.monotonic()
            heading_due = (
                now_monotonic
                - app.state.last_robot_heading_started_monotonic
                >= app.state.robot_heading_interval_seconds
            )

            if heading_due and heading_lock.acquire(blocking=False):
                try:
                    # Recheck after acquiring the lock in case another request
                    # completed a heading estimate between the first check and
                    # this acquisition.
                    now_monotonic = time.monotonic()
                    heading_due = (
                        now_monotonic
                        - app.state.last_robot_heading_started_monotonic
                        >= app.state.robot_heading_interval_seconds
                    )

                    if heading_due:
                        # Record the start time before inference. The heading
                        # lock itself prevents overlapping OA calls; this
                        # timestamp additionally caps requested starts at 10 Hz.
                        app.state.last_robot_heading_started_monotonic = (
                            now_monotonic
                        )

                        frame_rgb = cv2.cvtColor(
                            frame_bgr,
                            cv2.COLOR_BGR2RGB,
                        )

                        predict_kwargs: dict[str, Any] = {
                            "robot_box_xyxy": result.bbox_xyxy,
                        }

                        if (
                            camera_rotation is not None
                            and camera_intrinsics is not None
                        ):
                            predict_kwargs[
                                "camera_rotation_world_xyzw"
                            ] = (
                                camera_rotation["x"],
                                camera_rotation["y"],
                                camera_rotation["z"],
                                camera_rotation["w"],
                            )
                            predict_kwargs[
                                "camera_intrinsics"
                            ] = camera_intrinsics

                        try:
                            new_orientation_estimate = (
                                robot_forward_estimator.predict(
                                    frame_rgb,
                                    **predict_kwargs,
                                )
                            )

                            orientation_estimate = (
                                new_orientation_estimate
                            )
                            orientation_error = None

                            # Only publish the completed heading if the newest
                            # tracker state still has the robot. If tracking was
                            # lost while OA was running, do not resurrect a stale
                            # valid heading in shared state.
                            latest_tracking_result = (
                                app.state.latest_robot_tracking_result
                            )
                            if (
                                latest_tracking_result is not None
                                and latest_tracking_result.found
                            ):
                                app.state.latest_robot_orientation_estimate = (
                                    new_orientation_estimate
                                )
                                app.state.latest_robot_orientation_error = None

                        except Exception as exc:
                            # Keep the previous valid orientation estimate if
                            # one OA update fails. The error remains available
                            # diagnostically, while tracking can continue.
                            orientation_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            app.state.latest_robot_orientation_error = (
                                orientation_error
                            )
                            orientation_estimate = (
                                app.state.latest_robot_orientation_estimate
                            )

                            print(
                                "Robot forward estimation failed: "
                                f"{orientation_error}",
                                flush=True,
                            )

                finally:
                    heading_lock.release()

        else:
            # A cached heading may be retained internally for the next tracked
            # frame, but it must not be reported as valid while the current
            # tracker result says the robot is absent.
            orientation_estimate = None

        res = {
            "accepted": True,
            "reason": "processed",
            "submitted_frame_id": frame_id,
            "image_width": int(frame_bgr.shape[1]),
            "image_height": int(frame_bgr.shape[0]),
            "camera_pose_received": camera_position is not None,
            "camera_intrinsics_received": camera_intrinsics is not None,
            "result": format_robot_tracking_result(
                result,
                orientation_estimate,
                orientation_error,
            ),
        }
        print(res)

        return res


    #configure_posefix_for_hand_guidance()


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
        proj_joints, pose_img = construct_pose_img_from_joint_feature_vector(joint_feat_vec)
        quest_features = []
        if quest_model_features_type == "keypoints_projections":
            quest_features = np.concatenate([joint_feat_vec, proj_joints], axis=0)
        elif quest_model_features_type == "keypoints_resnet":
            resnet_embedding = resnet_encoder.predict(pose_img)["embedding"]
            quest_features = np.concatenate([joint_feat_vec, resnet_embedding.flatten().astype(np.float32)], axis=0)
        else:
            raise ValueError(
                f"Invalid quest_model_features_type: "
                f"{quest_model_features_type}"
            )

        probability_margin = 0.3

        target_probability = min(
            0.5 + probability_margin,
            1.0,
        )

        
        original_prediction = model.predict_features(
            quest_features
        )
        print("Original prediction:", original_prediction)

        target_intent = True

        if original_prediction.probability >= target_probability:
            print("NO PERTURBATIONS NEEDED")
            return {
                "perturbations_needed": False,
                "joint_position_perturbations": {},
                "joint_rotation_perturbations": {},
                "text_guidance": "",
            }
        pertstart = time.time()
        perturbed_result = find_minimal_perturbation(
            model=model,
            joint_feat_vec=quest_features,
            target_intent=target_intent,
            features_type=quest_model_features_type,
            original_probability=original_prediction.probability,
            classification_weight=1000.0,
            reachability_weight=1000.0,
            cross_body_weight= 1000.0,
            cross_body_margin=0.1,
            require_forward_extension_for_success=True,
            forward_extension_weight=1000.0,
            forward_extension_min=0.2,
            probability_margin=probability_margin,
            robustness_radius = 0.015,
            robustness_neighbor_margin = 0.15, 
            max_iterations=50,
            bin_search_iterations=5,
            object_arm=user_handedness
        )
        perturbation_success = perturbed_result["reached_target"]
        perturbed_features = perturbed_result["features"]
        perttotal = time.time() - pertstart
        print(f"Perturbation time: {perttotal:.3f} seconds")
        if(perturbation_success):
            print("PERTURBATION SUCCESSFUL")
        else:
            print("PERTURBATION FAILED, returning best effort")

        perturbed_joints = perturbed_features[:63]
        
        joint_feat_vec = joint_feat_vec[:-3]
        response = format_perturbation_response(
            original_joints=joint_feat_vec,
            perturbed_joints=perturbed_joints,
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
        camera_position_x: Optional[float] = Query(default=None),
        camera_position_y: Optional[float] = Query(default=None),
        camera_position_z: Optional[float] = Query(default=None),
        camera_rotation_x: Optional[float] = Query(default=None),
        camera_rotation_y: Optional[float] = Query(default=None),
        camera_rotation_z: Optional[float] = Query(default=None),
        camera_rotation_w: Optional[float] = Query(default=None),

        # Optional Quest RGB-camera intrinsics. These are NOT needed for the
        # signed relative-yaw estimate. They are only needed, together with
        # camera rotation, if the caller wants robot_forward_world_unit.
        camera_fx: Optional[float] = Query(default=None, gt=0.0),
        camera_fy: Optional[float] = Query(default=None, gt=0.0),
        camera_cx: Optional[float] = Query(default=None),
        camera_cy: Optional[float] = Query(default=None),
        camera_intrinsics_width: Optional[int] = Query(default=None, gt=0),
        camera_intrinsics_height: Optional[int] = Query(default=None, gt=0),
    ):
        """
        Receive one encoded Quest RGB image and return robot tracking data.

        Request body:
            Raw JPEG or PNG bytes.

        Required query parameter:
            frame_id

        Optional query parameters:
            capture_timestamp_unix

            camera_position_x
            camera_position_y
            camera_position_z
            camera_rotation_x
            camera_rotation_y
            camera_rotation_z
            camera_rotation_w

            camera_fx
            camera_fy
            camera_cx
            camera_cy
            camera_intrinsics_width
            camera_intrinsics_height

        Camera pose is optional, but when supplied all seven pose values must
        be present. Position is in the Quest world frame. Rotation is the
        CameraFrameCapture quaternion in Unity x/y/z/w order.

        Camera intrinsics are also optional. fx/fy/cx/cy must be supplied
        together. Calibration width/height may either both be supplied or
        both omitted. Intrinsics are only used to convert the visual relative
        yaw into a Quest-world forward unit vector; the relative-yaw model
        itself requires only the image and robot box.
        """
        camera_pose_values = [
            camera_position_x,
            camera_position_y,
            camera_position_z,
            camera_rotation_x,
            camera_rotation_y,
            camera_rotation_z,
            camera_rotation_w,
        ]

        any_camera_pose_value = any(
            value is not None
            for value in camera_pose_values
        )

        all_camera_pose_values = all(
            value is not None
            for value in camera_pose_values
        )

        if any_camera_pose_value and not all_camera_pose_values:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Camera pose was only partially supplied. "
                    "Provide all camera_position_x/y/z and "
                    "camera_rotation_x/y/z/w values, or omit "
                    "the camera pose entirely."
                ),
            )

        camera_position = None
        camera_rotation = None

        if all_camera_pose_values:
            camera_position = {
                "x": float(camera_position_x),
                "y": float(camera_position_y),
                "z": float(camera_position_z),
            }

            camera_rotation = {
                "x": float(camera_rotation_x),
                "y": float(camera_rotation_y),
                "z": float(camera_rotation_z),
                "w": float(camera_rotation_w),
            }

        intrinsic_core_values = [
            camera_fx,
            camera_fy,
            camera_cx,
            camera_cy,
        ]
        any_intrinsic_core = any(
            value is not None
            for value in intrinsic_core_values
        )
        all_intrinsic_core = all(
            value is not None
            for value in intrinsic_core_values
        )

        if any_intrinsic_core and not all_intrinsic_core:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Camera intrinsics were only partially supplied. "
                    "Provide camera_fx/fy/cx/cy together, or omit all four."
                ),
            )

        any_intrinsic_size = (
            camera_intrinsics_width is not None
            or camera_intrinsics_height is not None
        )
        all_intrinsic_size = (
            camera_intrinsics_width is not None
            and camera_intrinsics_height is not None
        )

        if any_intrinsic_size and not all_intrinsic_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    "camera_intrinsics_width and "
                    "camera_intrinsics_height must be supplied together."
                ),
            )

        if all_intrinsic_size and not all_intrinsic_core:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Intrinsics calibration resolution cannot be supplied "
                    "without camera_fx/fy/cx/cy."
                ),
            )

        if all_intrinsic_core and camera_rotation is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Camera intrinsics require the full camera pose because "
                    "camera_rotation_x/y/z/w is needed to express the "
                    "estimated robot forward direction in Quest world."
                ),
            )

        camera_intrinsics = None

        if all_intrinsic_core:
            camera_intrinsics = {
                "fx": float(camera_fx),
                "fy": float(camera_fy),
                "cx": float(camera_cx),
                "cy": float(camera_cy),
            }

            if all_intrinsic_size:
                camera_intrinsics["width"] = int(camera_intrinsics_width)
                camera_intrinsics["height"] = int(camera_intrinsics_height)

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
            camera_position,
            camera_rotation,
            camera_intrinsics,
        )

    # Optional endpoint for forcing a clean reacquisition.
    @app.post("/reset_robot_tracker")
    def reset_robot_tracker():
        tracking_lock: threading.Lock = (
            app.state.robot_tracking_lock
        )
        heading_lock: threading.Lock = (
            app.state.robot_heading_lock
        )

        # process_robot_frame never holds these two locks simultaneously:
        # tracking is released before heading is acquired. Acquiring both here
        # prevents an in-flight heading estimate from writing stale state after
        # the reset completes.
        with tracking_lock:
            with heading_lock:
                app.state.robot_tracker.reset()
                app.state.latest_robot_tracking_result = None
                app.state.latest_robot_orientation_estimate = None
                app.state.latest_robot_orientation_error = None
                app.state.last_robot_frame_id = -1
                app.state.last_robot_heading_started_monotonic = float("-inf")
                app.state.latest_robot_camera_position = None
                app.state.latest_robot_camera_rotation = None
                app.state.latest_robot_camera_intrinsics = None

        return {
            "success": True,
            "state": "searching",
        }

    return app
        



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quest-model-type",
        type=str,
        choices=QUEST_MODEL_TYPES,
        default="tabm",
        help=(
            "Quest student model architecture. Defaults to tabm."
        ),
    )

    parser.add_argument(
        "--quest-model-features-type",
        type=str,
        choices=QUEST_MODEL_FEATURES_TYPES,
        default="keypoints_projections",
        help=(
            "Quest student feature representation. Defaults to "
            "keypoints_projections."
        ),
    )

    parser.add_argument(
        "--quest-estimator-path",
        type=str,
        default=None,
        help=(
            "Optional explicit Quest estimator checkpoint path. If supplied, "
            "this overrides automatic path resolution from --quest-model-type "
            "and --quest-model-features-type."
        ),
    )

    parser.add_argument(
            "--user-handedness",
            type=str,
            default=None,
            help=(
                "Optional explicit user handedness for counterfactual optimization. If supplied, this will make optimization run for only one arm, assuming this is the most likely arm for an object to be held in. Defaults to both."
            ),
        )

    parser.add_argument(
        "--robot-obj-det-weights-path",
        type=str,
        default=None,
        help=(
            "Optional explicit Go2 YOLO weights. When supplied, the same "
            "weights are used by both RobotDetectorTracker and the forward "
            "estimator. If omitted, the forward-estimation checkpoint's "
            "saved/default YOLO configuration is used."
        ),
    )

    parser.add_argument(
        "--robot-forward-estimator-path",
        type=str,
        default=None,
        help=(
            "Optional explicit final all-data Orient Anything forward-"
            "estimation checkpoint. If omitted, the deployment wrapper uses "
            "its normal default under "
            "robot_fov_estimation/outputs/"
            "orient_anything_forward_estimation_weights/."
        ),
    )

    parser.add_argument(
        "--robot-forward-decoder",
        type=str,
        choices=("mean", "argmax"),
        default="mean",
        help=(
            "Decoder for signed relative yaw. Defaults to circular mean."
        ),
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

    quest_estimator_path = resolve_quest_estimator_path(
        quest_model_type=args.quest_model_type,
        quest_model_features_type=args.quest_model_features_type,
        override_path=args.quest_estimator_path,
    )

    print(
        "Quest model configuration: "
        f"type={args.quest_model_type}, "
        f"features={args.quest_model_features_type}, "
        f"checkpoint={quest_estimator_path}"
    )

    app = create_app(
        args.robot_obj_det_weights_path,
        str(quest_estimator_path),
        args.quest_model_type,
        args.quest_model_features_type,
        args.robot_forward_estimator_path,
        args.robot_forward_decoder,
        user_handedness=args.user_handedness
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()