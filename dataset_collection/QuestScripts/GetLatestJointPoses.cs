using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public struct UpperBodyJointIds
{
    public OVRSkeleton.BoneId Root;
    public OVRSkeleton.BoneId Hips;

    public OVRSkeleton.BoneId SpineLower;
    public OVRSkeleton.BoneId SpineMiddle;
    public OVRSkeleton.BoneId SpineUpper;
    public OVRSkeleton.BoneId Chest;
    public OVRSkeleton.BoneId Neck;
    public OVRSkeleton.BoneId Head;

    public OVRSkeleton.BoneId LeftShoulder;
    public OVRSkeleton.BoneId RightShoulder;

    public OVRSkeleton.BoneId LeftScapula;
    public OVRSkeleton.BoneId RightScapula;

    public OVRSkeleton.BoneId LeftUpperArm;
    public OVRSkeleton.BoneId RightUpperArm;

    public OVRSkeleton.BoneId LeftForearm;
    public OVRSkeleton.BoneId RightForearm;

    // Original wrist joints you were already using.
    public OVRSkeleton.BoneId LeftHand;
    public OVRSkeleton.BoneId RightHand;

    // Extra hand/wrist joints.
    public OVRSkeleton.BoneId LeftWristTwist;
    public OVRSkeleton.BoneId RightWristTwist;

    public OVRSkeleton.BoneId LeftPalm;
    public OVRSkeleton.BoneId RightPalm;

    public static UpperBodyJointIds Default => new UpperBodyJointIds
    {
        Root = OVRSkeleton.BoneId.Body_Root,
        Hips = OVRSkeleton.BoneId.Body_Hips,

        SpineLower = OVRSkeleton.BoneId.Body_SpineLower,
        SpineMiddle = OVRSkeleton.BoneId.Body_SpineMiddle,
        SpineUpper = OVRSkeleton.BoneId.Body_SpineUpper,
        Chest = OVRSkeleton.BoneId.Body_Chest,
        Neck = OVRSkeleton.BoneId.Body_Neck,
        Head = OVRSkeleton.BoneId.Body_Head,

        LeftShoulder = OVRSkeleton.BoneId.Body_LeftShoulder,
        RightShoulder = OVRSkeleton.BoneId.Body_RightShoulder,

        LeftScapula = OVRSkeleton.BoneId.Body_LeftScapula,
        RightScapula = OVRSkeleton.BoneId.Body_RightScapula,

        LeftUpperArm = OVRSkeleton.BoneId.Body_LeftArmUpper,
        RightUpperArm = OVRSkeleton.BoneId.Body_RightArmUpper,

        LeftForearm = OVRSkeleton.BoneId.Body_LeftArmLower,
        RightForearm = OVRSkeleton.BoneId.Body_RightArmLower,

        LeftHand = OVRSkeleton.BoneId.Body_LeftHandWrist,
        RightHand = OVRSkeleton.BoneId.Body_RightHandWrist,

        LeftWristTwist = OVRSkeleton.BoneId.Body_LeftHandWristTwist,
        RightWristTwist = OVRSkeleton.BoneId.Body_RightHandWristTwist,

        LeftPalm = OVRSkeleton.BoneId.Body_LeftHandPalm,
        RightPalm = OVRSkeleton.BoneId.Body_RightHandPalm
    };
}

[Serializable]
public struct JointPoseData
{
    public string jointName;
    public OVRSkeleton.BoneId boneId;

    public bool valid;

    // Unity world-space pose.
    public Vector3 position;
    public Quaternion rotation;

    // Convenience version if another script wants Euler angles.
    public Vector3 eulerAngles;

    public float unityTime;
    public int unityFrame;
}

[Serializable]
public class UpperBodyJointSnapshot
{
    public float unityTime;
    public int unityFrame;
    public string coordinateFrame = "unity_world";

    // True headset pose, if centerEyeAnchor is assigned.
    // This is not an OVRSkeleton body bone, but is useful for trajectory prediction.
    public JointPoseData hmdCenterEye;

    public JointPoseData root;
    public JointPoseData hips;

    public JointPoseData spineLower;
    public JointPoseData spineMiddle;
    public JointPoseData spineUpper;
    public JointPoseData chest;
    public JointPoseData neck;
    public JointPoseData head;

    public JointPoseData leftShoulder;
    public JointPoseData rightShoulder;

    public JointPoseData leftScapula;
    public JointPoseData rightScapula;

    public JointPoseData leftUpperArm;
    public JointPoseData rightUpperArm;

    public JointPoseData leftForearm;
    public JointPoseData rightForearm;

    // Original wrist joints.
    public JointPoseData leftHand;
    public JointPoseData rightHand;

    // Extra wrist/hand joints.
    public JointPoseData leftWristTwist;
    public JointPoseData rightWristTwist;

    public JointPoseData leftPalm;
    public JointPoseData rightPalm;
}

public class GetLatestJointPoses : MonoBehaviour
{
    [Header("OVR References")]
    public OVRSkeleton skeleton;

    [Tooltip("Optional. Usually OVRCameraRig.centerEyeAnchor. This gives the true tracked HMD/headset pose.")]
    public Transform centerEyeAnchor;

    [Header("Joint IDs")]
    public UpperBodyJointIds jointIds = UpperBodyJointIds.Default;

    [Header("Data Quality")]
    public bool requireHighConfidence = true;

    public bool HasValidSnapshot { get; private set; }

    public UpperBodyJointSnapshot LatestSnapshot { get; private set; } =
        new UpperBodyJointSnapshot();

    private readonly Dictionary<string, JointPoseData> latestByName =
        new Dictionary<string, JointPoseData>();

    private void Awake()
    {
        if (skeleton == null)
        {
            skeleton = GetComponentInChildren<OVRSkeleton>();
        }

        // Optional convenience fallback.
        // You can also just assign centerEyeAnchor manually in the Inspector.
        if (centerEyeAnchor == null)
        {
            OVRCameraRig cameraRig = FindObjectOfType<OVRCameraRig>();

            if (cameraRig != null)
            {
                centerEyeAnchor = cameraRig.centerEyeAnchor;
            }
        }
    }

    private void Update()
    {
        UpdateLatestJointPoses();
    }

    private void UpdateLatestJointPoses()
    {
        float now = Time.time;
        int frame = Time.frameCount;

        latestByName.Clear();

        LatestSnapshot = new UpperBodyJointSnapshot
        {
            unityTime = now,
            unityFrame = frame,
            coordinateFrame = "unity_world"
        };

        // This is independent of OVRSkeleton body tracking.
        // It is the actual headset/center-eye pose if available.
        LatestSnapshot.hmdCenterEye = ReadTransformPose(
            "hmd_center_eye",
            centerEyeAnchor,
            now,
            frame
        );

        if (skeleton == null ||
            !skeleton.IsInitialized ||
            !skeleton.IsDataValid ||
            skeleton.Bones == null ||
            skeleton.Bones.Count == 0)
        {
            HasValidSnapshot = LatestSnapshot.hmdCenterEye.valid;
            return;
        }

        // Optional but recommended for clean dataset collection.
        if (requireHighConfidence && !skeleton.IsDataHighConfidence)
        {
            HasValidSnapshot = LatestSnapshot.hmdCenterEye.valid;
            return;
        }

        LatestSnapshot.root = ReadJoint("root", jointIds.Root, now, frame);
        LatestSnapshot.hips = ReadJoint("hips", jointIds.Hips, now, frame);

        LatestSnapshot.spineLower = ReadJoint("spine_lower", jointIds.SpineLower, now, frame);
        LatestSnapshot.spineMiddle = ReadJoint("spine_middle", jointIds.SpineMiddle, now, frame);
        LatestSnapshot.spineUpper = ReadJoint("spine_upper", jointIds.SpineUpper, now, frame);
        LatestSnapshot.chest = ReadJoint("chest", jointIds.Chest, now, frame);
        LatestSnapshot.neck = ReadJoint("neck", jointIds.Neck, now, frame);
        LatestSnapshot.head = ReadJoint("head", jointIds.Head, now, frame);

        LatestSnapshot.leftShoulder = ReadJoint("left_shoulder", jointIds.LeftShoulder, now, frame);
        LatestSnapshot.rightShoulder = ReadJoint("right_shoulder", jointIds.RightShoulder, now, frame);

        LatestSnapshot.leftScapula = ReadJoint("left_scapula", jointIds.LeftScapula, now, frame);
        LatestSnapshot.rightScapula = ReadJoint("right_scapula", jointIds.RightScapula, now, frame);

        LatestSnapshot.leftUpperArm = ReadJoint("left_upper_arm", jointIds.LeftUpperArm, now, frame);
        LatestSnapshot.rightUpperArm = ReadJoint("right_upper_arm", jointIds.RightUpperArm, now, frame);

        LatestSnapshot.leftForearm = ReadJoint("left_forearm", jointIds.LeftForearm, now, frame);
        LatestSnapshot.rightForearm = ReadJoint("right_forearm", jointIds.RightForearm, now, frame);

        LatestSnapshot.leftHand = ReadJoint("left_hand", jointIds.LeftHand, now, frame);
        LatestSnapshot.rightHand = ReadJoint("right_hand", jointIds.RightHand, now, frame);

        LatestSnapshot.leftWristTwist = ReadJoint("left_wrist_twist", jointIds.LeftWristTwist, now, frame);
        LatestSnapshot.rightWristTwist = ReadJoint("right_wrist_twist", jointIds.RightWristTwist, now, frame);

        LatestSnapshot.leftPalm = ReadJoint("left_palm", jointIds.LeftPalm, now, frame);
        LatestSnapshot.rightPalm = ReadJoint("right_palm", jointIds.RightPalm, now, frame);

        HasValidSnapshot =
            LatestSnapshot.hmdCenterEye.valid ||
            LatestSnapshot.head.valid ||
            LatestSnapshot.neck.valid ||
            LatestSnapshot.chest.valid ||
            LatestSnapshot.hips.valid ||
            LatestSnapshot.leftHand.valid ||
            LatestSnapshot.rightHand.valid ||
            LatestSnapshot.leftPalm.valid ||
            LatestSnapshot.rightPalm.valid;
    }

    private JointPoseData ReadJoint(
        string jointName,
        OVRSkeleton.BoneId boneId,
        float unityTime,
        int unityFrame)
    {
        JointPoseData data = new JointPoseData
        {
            jointName = jointName,
            boneId = boneId,
            valid = false,
            position = Vector3.zero,
            rotation = Quaternion.identity,
            eulerAngles = Vector3.zero,
            unityTime = unityTime,
            unityFrame = unityFrame
        };

        if (TryGetWorldPose(skeleton, boneId, out Vector3 position, out Quaternion rotation))
        {
            data.valid = true;
            data.position = position;
            data.rotation = rotation;
            data.eulerAngles = rotation.eulerAngles;
        }

        latestByName[jointName] = data;
        return data;
    }

    private JointPoseData ReadTransformPose(
        string jointName,
        Transform source,
        float unityTime,
        int unityFrame)
    {
        JointPoseData data = new JointPoseData
        {
            jointName = jointName,

            // The HMD/center-eye pose is not actually an OVRSkeleton bone.
            // Invalid is used here as a sentinel.
            boneId = OVRSkeleton.BoneId.Invalid,

            valid = false,
            position = Vector3.zero,
            rotation = Quaternion.identity,
            eulerAngles = Vector3.zero,
            unityTime = unityTime,
            unityFrame = unityFrame
        };

        if (source != null)
        {
            data.valid = true;
            data.position = source.position;
            data.rotation = source.rotation;
            data.eulerAngles = source.rotation.eulerAngles;
        }

        latestByName[jointName] = data;
        return data;
    }

    public static bool TryGetWorldPose(
        OVRSkeleton skeleton,
        OVRSkeleton.BoneId joint,
        out Vector3 position,
        out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;

        if (skeleton == null)
            return false;

        var bones = skeleton.Bones;

        if (bones == null || bones.Count == 0)
            return false;

        for (int i = 0; i < bones.Count; i++)
        {
            var bone = bones[i];

            if (bone.Id == joint && bone.Transform != null)
            {
                Transform t = bone.Transform;

                position = t.position;
                rotation = t.rotation;

                return true;
            }
        }

        return false;
    }

    public bool TryGetJointPose(string jointName, out JointPoseData pose)
    {
        return latestByName.TryGetValue(jointName, out pose) && pose.valid;
    }

    public bool TryGetJointPose(
        OVRSkeleton.BoneId boneId,
        out JointPoseData pose)
    {
        foreach (var kvp in latestByName)
        {
            if (kvp.Value.boneId == boneId && kvp.Value.valid)
            {
                pose = kvp.Value;
                return true;
            }
        }

        pose = default;
        return false;
    }

    public UpperBodyJointSnapshot GetLatestSnapshot()
    {
        return LatestSnapshot;
    }

    public void DumpAvailableBones()
    {
        if (skeleton == null ||
            !skeleton.IsInitialized ||
            skeleton.Bones == null ||
            skeleton.Bones.Count == 0)
        {
            Debug.LogWarning("No valid OVRSkeleton bones available to dump.");
            return;
        }

        Debug.Log($"OVRSkeleton has {skeleton.Bones.Count} bones.");

        for (int i = 0; i < skeleton.Bones.Count; i++)
        {
            var bone = skeleton.Bones[i];

            if (bone.Transform == null)
                continue;

            Debug.Log(
                $"{i}: {bone.Id} | pos={bone.Transform.position} | rot={bone.Transform.rotation.eulerAngles}"
            );
        }
    }
}