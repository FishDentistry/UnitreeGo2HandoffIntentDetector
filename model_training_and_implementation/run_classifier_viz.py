import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.dino_detector import DINOObjectDetector
from src.rtmpose_keypoints import RTMPoseKeypointDetector
from src.head_pose import KwanHeadPoseEstimator
from src.kwan_classifier import KwanLocalizedMLPClassifier
from shared.util.extract_all_samples import discover_samples, sample_root_from_dataset_root, read_json, Sample

# Optional: use Kwan's original head-pose drawing utilities if copied over.
try:
    from src.kwan_headpose.utils import draw_axis, plot_pose_cube
except Exception:
    draw_axis = None
    plot_pose_cube = None


COCO_SKELETON = [
    (0, 1), (0, 2),        # nose-eyes
    (1, 3), (2, 4),        # eyes-ears
    (5, 6),                # shoulders
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 11), (6, 12),      # torso
    (11, 12),              # hips
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
]

LEFT_WRIST = 9
RIGHT_WRIST = 10


def _get_attr_or_key(x, name, default=None):
    if isinstance(x, dict):
        return x.get(name, default)
    return getattr(x, name, default)


def get_box_xyxy(obj):
    box = _get_attr_or_key(obj, "box_xyxy", None)
    if box is None:
        box = _get_attr_or_key(obj, "bbox", None)
    if box is None:
        box = _get_attr_or_key(obj, "box", None)
    if box is None:
        raise ValueError(f"Could not find box field in object: {obj}")
    return np.asarray(box, dtype=np.float32)


def get_score(obj):
    score = _get_attr_or_key(obj, "score", None)
    if score is None:
        return None
    return float(score)


def get_class_name(obj):
    class_name = _get_attr_or_key(obj, "class_name", None)
    if class_name is None:
        class_name = _get_attr_or_key(obj, "label", None)
    if class_name is None:
        class_name = _get_attr_or_key(obj, "name", None)
    if class_name is None:
        class_name = "object"
    return str(class_name)


def get_keypoints(person):
    keypoints = _get_attr_or_key(person, "keypoints", None)
    if keypoints is None:
        keypoints = _get_attr_or_key(person, "keypoints_xy", None)
    if keypoints is None:
        keypoints = _get_attr_or_key(person, "pred_keypoints", None)
    if keypoints is None:
        raise ValueError(f"Could not find keypoints in person: {person}")

    keypoints = np.asarray(keypoints, dtype=np.float32)

    # Accept either [17, 2] or [17, 3].
    if keypoints.shape[1] == 2:
        conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
        keypoints = np.concatenate([keypoints, conf], axis=1)

    return keypoints


def box_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def draw_label(img, text, xy, bg=(0, 0, 0), fg=(255, 255, 255)):
    x, y = int(xy[0]), int(xy[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y), bg, -1)
    cv2.putText(img, text, (x + 2, y - baseline - 2), font, scale, fg, thickness, cv2.LINE_AA)


def draw_objects(img, objects, selected_object_index=None):
    for i, obj in enumerate(objects):
        box = get_box_xyxy(obj).astype(int)
        score = get_score(obj)
        name = get_class_name(obj)

        selected = selected_object_index is not None and i == selected_object_index
        color = (0, 255, 255) if selected else (0, 180, 0)
        thickness = 3 if selected else 2

        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if score is None:
            label = f"{i}: {name}"
        else:
            label = f"{i}: {name} {score:.2f}"

        if selected:
            label = "SELECTED " + label

        draw_label(img, label, (x1, max(y1 - 2, 15)), bg=color, fg=(0, 0, 0))


def draw_people(img, people, selected_person_index=None, kp_conf_thresh=0.05):
    for i, person in enumerate(people):
        kpts = get_keypoints(person)

        selected = selected_person_index is not None and i == selected_person_index
        color = (255, 0, 255) if selected else (255, 160, 0)
        joint_color = (0, 0, 255) if selected else (255, 0, 0)
        thickness = 3 if selected else 2

        # Draw skeleton.
        for a, b in COCO_SKELETON:
            if a >= len(kpts) or b >= len(kpts):
                continue
            if kpts[a, 2] < kp_conf_thresh or kpts[b, 2] < kp_conf_thresh:
                continue

            xa, ya = int(kpts[a, 0]), int(kpts[a, 1])
            xb, yb = int(kpts[b, 0]), int(kpts[b, 1])
            cv2.line(img, (xa, ya), (xb, yb), color, thickness)

        # Draw joints.
        for j, kp in enumerate(kpts):
            x, y, conf = kp[:3]
            if conf < kp_conf_thresh:
                continue

            radius = 5 if selected else 4
            cv2.circle(img, (int(x), int(y)), radius, joint_color, -1)

        # Label near nose if available.
        if len(kpts) > 0:
            nose = kpts[0]
            if nose[2] >= kp_conf_thresh:
                text = f"person {i}"
                if selected:
                    text = "SELECTED " + text
                draw_label(img, text, (nose[0], nose[1] - 10), bg=color, fg=(0, 0, 0))


def draw_association(img, objects, people, association):
    if not association:
        return

    person_idx = association.get("selected_person_index")
    object_idx = association.get("selected_object_index")
    wrist_name = association.get("selected_wrist")

    if person_idx is None or object_idx is None:
        return

    if person_idx >= len(people) or object_idx >= len(objects):
        return

    kpts = get_keypoints(people[person_idx])
    obj_box = get_box_xyxy(objects[object_idx])
    obj_center = box_center(obj_box)

    wrist_idx = LEFT_WRIST if wrist_name == "left_wrist" else RIGHT_WRIST
    wrist_xy = kpts[wrist_idx, 0:2]

    p1 = (int(wrist_xy[0]), int(wrist_xy[1]))
    p2 = (int(obj_center[0]), int(obj_center[1]))

    cv2.line(img, p1, p2, (0, 255, 255), 3)
    cv2.circle(img, p1, 8, (0, 255, 255), -1)
    cv2.circle(img, p2, 8, (0, 255, 255), -1)

    dist = association.get("wrist_object_distance_px")
    if dist is not None:
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        draw_label(img, f"{wrist_name} -> object: {dist:.1f}px", mid, bg=(0, 255, 255), fg=(0, 0, 0))


def draw_head_pose(img, head_pose_results, fallback_head_pose=None):
    """
    Draws the first detected head pose. If Kwan's plot_pose_cube is available,
    uses it. Otherwise draws text only.
    """

    if head_pose_results:
        hp = head_pose_results[0]

        yaw = float(_get_attr_or_key(hp, "yaw"))
        pitch = float(_get_attr_or_key(hp, "pitch"))
        roll = float(_get_attr_or_key(hp, "roll"))

        box = _get_attr_or_key(hp, "box_xyxy", None)
        if box is not None:
            box = np.asarray(box, dtype=np.float32)
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = box_center(box)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            draw_label(img, f"head yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}", (x1, y1 - 2), bg=(255, 255, 0), fg=(0, 0, 0))

            if plot_pose_cube is not None:
                try:
                    plot_pose_cube(img, yaw, pitch, roll, tdx=float(cx), tdy=float(cy), size=80)
                    return
                except Exception:
                    pass

            if draw_axis is not None:
                try:
                    draw_axis(img, yaw, pitch, roll, tdx=float(cx), tdy=float(cy), size=80)
                    return
                except Exception:
                    pass

        else:
            draw_label(img, f"head yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}", (20, 90), bg=(255, 255, 0), fg=(0, 0, 0))

    elif fallback_head_pose is not None:
        hp = np.asarray(fallback_head_pose).flatten()
        draw_label(img, f"head pose: [{hp[0]:.1f}, {hp[1]:.1f}, {hp[2]:.1f}]", (20, 90), bg=(80, 80, 80), fg=(255, 255, 255))


def draw_prediction_banner(img, result):
    score = float(result.get("handoff_score", 0.0))
    detected = bool(result.get("handoff_detected", False))
    threshold = float(result.get("threshold", 0.5))

    if detected:
        text = f"HANDOFF DETECTED  score={score:.3f}  threshold={threshold:.2f}"
        bg = (0, 200, 0)
    else:
        text = f"NO HANDOFF  score={score:.3f}  threshold={threshold:.2f}"
        bg = (0, 0, 220)

    cv2.rectangle(img, (0, 0), (img.shape[1], 45), bg, -1)
    cv2.putText(
        img,
        text,
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_output_image_path(image_path, output_image_arg):
    if output_image_arg:
        return output_image_arg

    image_path = Path(image_path)
    out_dir = Path("outputs/debug_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{image_path.stem}_result.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", default="outputs/predictions.json")
    parser.add_argument("--output-image", default=None)
    parser.add_argument("--dino-classes", default="cup")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--mlp-weights", default="kwan_pretrained_weights/MLP_localized.pth")
    parser.add_argument("--head-pose-weights", default="kwan_pretrained_weights/head-pose-pretrained.pkl")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")



    object_detector = DINOObjectDetector(
        model_id="IDEA-Research/grounding-dino-base",
        confidence=args.confidence,
    )

    keypoint_detector = RTMPoseKeypointDetector(
        confidence=args.confidence,
        device="cuda",
    )

    head_pose_estimator = KwanHeadPoseEstimator(
        weights_path=args.head_pose_weights,
    )

    classifier = KwanLocalizedMLPClassifier(
        weights_path=args.mlp_weights,
        threshold=0.5,
    )

    objects = object_detector.predict(image, class_names=args.dino_classes)
    people = keypoint_detector.predict(image)

    # Use full head pose result for drawing, but pass only [yaw, pitch, roll] to classifier.
    head_pose_results = head_pose_estimator.predict(image)
    if head_pose_results:
        print("HP RES")
        print(head_pose_results)
        hp0 = head_pose_results[0]
        head_pose = np.asarray([hp0.yaw, hp0.pitch, hp0.roll], dtype=np.float32)
    else:
        head_pose = np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)
    
    #head_pose = np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)

    result = classifier.predict(
        objects=objects,
        people=people,
        head_pose=head_pose,
    )

    result["image"] = args.image
    result["num_objects"] = len(objects)
    result["num_people"] = len(people)
    result["head_pose"] = head_pose.tolist()

    # Save JSON.
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    # Draw debug/result image.
    vis = image.copy()

    association = result.get("association", {})
    selected_object_index = association.get("selected_object_index")
    selected_person_index = association.get("selected_person_index")

    draw_objects(vis, objects, selected_object_index=selected_object_index)
    draw_people(vis, people, selected_person_index=selected_person_index)
    draw_association(vis, objects, people, association)
    draw_head_pose(vis, head_pose_results, fallback_head_pose=head_pose)
    draw_prediction_banner(vis, result)

    output_image = make_output_image_path(args.image, args.output_image)
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    cv2.imwrite(output_image, vis)

    print(json.dumps(result, indent=2))
    print(f"\nWrote result image: {output_image}")
    print(f"Wrote result json:  {args.output_json}")



if __name__ == "__main__":
    main()