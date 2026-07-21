
#Run with: python -m servers.check_current_handoff_pose_server --quest-mlp-est-path /path/to/mlp_weights_file/file.pth
import argparse
import uvicorn
from fastapi import FastAPI
from typing import Any
import numpy as np

from quest_hand_intent_model_est.src.quest_hand_int_inf_wrapper import (
    QuestHandIntentEstInference,
)
from quest_hand_intent_model_est.src.quest_joint_features import (
    extract_quest_joint_features_json, feature_vector_to_joint_poses
)
from quest_hand_intent_model_est.src.find_minimal_perturbation import find_minimal_perturbation
from shared.util.quest_joints_network_dat_str import QuestJointsPacket


def format_perturbation_response(
    perturbed_feature_vector: np.ndarray,
    perturbations_needed: bool = True,
) -> dict[str, Any]:
    """
    Convert the output of find_minimal_perturbation() into the response
    format expected by Unity.

    Positions are final joint positions relative to the shoulder midpoint.

    When rotations are present in the feature vector, they are included in
    the response. When the model uses position-only features, the rotation
    dictionary remains empty.
    """
    joint_poses = feature_vector_to_joint_poses(
        perturbed_feature_vector
    )

    joint_position_perturbations: dict[
        str,
        list[float],
    ] = {}

    joint_rotation_perturbations: dict[
        str,
        list[float],
    ] = {}

    for joint_name, joint_pose in joint_poses.items():
        position = joint_pose["position"]

        joint_position_perturbations[joint_name] = (
            position.astype(float).tolist()
        )

        rotation = joint_pose["rotation"]

        if rotation is not None:
            joint_rotation_perturbations[joint_name] = (
                rotation.astype(float).tolist()
            )
    print(joint_position_perturbations)
    return {
        "perturbations_needed": perturbations_needed,
        "joint_position_perturbations":
            joint_position_perturbations,
        "joint_rotation_perturbations":
            joint_rotation_perturbations,
    }


def create_app(model_path: str) -> FastAPI:
    app = FastAPI()

    model = QuestHandIntentEstInference(
        checkpoint_path=model_path,
    )

    @app.post("/quest_joints")
    def receive_quest_joints(packet: QuestJointsPacket):
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
            }

        perturbed_features = find_minimal_perturbation(
            model=model,
            joint_feat_vec=joint_feat_vec,
            target_intent=True,
            classification_weight=1000.0,
            bone_length_weight=1000.0,
        )

        response = format_perturbation_response(
            perturbed_feature_vector=perturbed_features,
            perturbations_needed=True,
        )

        # Convert shoulder-relative positions back into Unity world positions.
        for position in response[
            "joint_position_perturbations"
        ].values():
            position[0] += shoulder_midpoint[0]
            position[1] += shoulder_midpoint[1]
            position[2] += shoulder_midpoint[2]

        return response
        

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

    app = create_app(args.quest_mlp_est_path)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()