using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class SendQuestJointsToServer : MonoBehaviour
{
    [Header("Joint Provider")]
    public GetLatestJointPoses jointProvider;

    [Header("Server")]
    public string serverUrl = "http://YOUR_SERVER_IP:8000/quest_joints";

    [Header("Sending")]
    public float sendRateHz = 30.0f;
    public bool sendAutomatically = true;
    public bool printDebugLogs = false;

    private Coroutine sendCoroutine;

    private void Awake()
    {
        if (jointProvider == null)
        {
            jointProvider = GetComponent<GetLatestJointPoses>();
        }

        if (jointProvider == null)
        {
            jointProvider = GetComponentInChildren<GetLatestJointPoses>();
        }
    }

    private void OnEnable()
    {
        if (sendAutomatically)
        {
            StartSending();
        }
    }

    private void OnDisable()
    {
        StopSending();
    }

    public void StartSending()
    {
        if (sendCoroutine != null)
            return;

        sendCoroutine = StartCoroutine(SendLoop());
    }

    public void StopSending()
    {
        if (sendCoroutine != null)
        {
            StopCoroutine(sendCoroutine);
            sendCoroutine = null;
        }
    }

    private IEnumerator SendLoop()
    {
        float waitSeconds = 1.0f / Mathf.Max(sendRateHz, 1.0f);
        WaitForSeconds wait = new WaitForSeconds(waitSeconds);

        while (true)
        {
            yield return SendLatestJoints();
            yield return wait;
        }
    }

    public IEnumerator SendLatestJoints()
    {
        if (jointProvider == null)
        {
            Debug.LogWarning("[SendQuestJointsToServer] No joint provider assigned.");
            yield break;
        }

        if (!jointProvider.HasValidSnapshot)
        {
            if (printDebugLogs)
                Debug.Log("[SendQuestJointsToServer] No valid joint snapshot yet.");

            yield break;
        }

        UpperBodyJointSnapshot snapshot = jointProvider.GetLatestSnapshot();

        QuestJointPacket packet = BuildPacket(snapshot);
        string json = JsonUtility.ToJson(packet);

        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest request = new UnityWebRequest(serverUrl, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning(
                    "[SendQuestJointsToServer] Failed to send joints: " +
                    request.error +
                    "\nURL: " + serverUrl
                );
            }
            else if (printDebugLogs)
            {
                Debug.Log("[SendQuestJointsToServer] Sent joints: " + json);
            }
        }
    }

    private QuestJointPacket BuildPacket(UpperBodyJointSnapshot snapshot)
    {
        return new QuestJointPacket
        {
            quest_time_ns = GetUnixTimeNanoseconds(),

            unity_time = snapshot.unityTime,
            frame_id = snapshot.unityFrame,
            coordinate_frame = snapshot.coordinateFrame,

            joints = new QuestUpperBodyJoints
            {
                // Actual HMD / center-eye pose, if assigned in GetLatestJointPoses.
                hmd_center_eye = ConvertJoint(snapshot.hmdCenterEye),

                root = ConvertJoint(snapshot.root),
                hips = ConvertJoint(snapshot.hips),

                spine_lower = ConvertJoint(snapshot.spineLower),
                spine_middle = ConvertJoint(snapshot.spineMiddle),
                spine_upper = ConvertJoint(snapshot.spineUpper),
                chest = ConvertJoint(snapshot.chest),
                neck = ConvertJoint(snapshot.neck),
                head = ConvertJoint(snapshot.head),

                left_shoulder = ConvertJoint(snapshot.leftShoulder),
                right_shoulder = ConvertJoint(snapshot.rightShoulder),

                left_scapula = ConvertJoint(snapshot.leftScapula),
                right_scapula = ConvertJoint(snapshot.rightScapula),

                left_upper_arm = ConvertJoint(snapshot.leftUpperArm),
                right_upper_arm = ConvertJoint(snapshot.rightUpperArm),

                left_forearm = ConvertJoint(snapshot.leftForearm),
                right_forearm = ConvertJoint(snapshot.rightForearm),

                left_hand = ConvertJoint(snapshot.leftHand),
                right_hand = ConvertJoint(snapshot.rightHand),

                left_wrist_twist = ConvertJoint(snapshot.leftWristTwist),
                right_wrist_twist = ConvertJoint(snapshot.rightWristTwist),

                left_palm = ConvertJoint(snapshot.leftPalm),
                right_palm = ConvertJoint(snapshot.rightPalm)
            }
        };
    }

    private QuestJointData ConvertJoint(JointPoseData joint)
    {
        return new QuestJointData
        {
            joint_name = joint.jointName,
            bone_id = joint.boneId.ToString(),

            valid = joint.valid,

            position = new float[]
            {
                joint.position.x,
                joint.position.y,
                joint.position.z
            },

            rotation = new float[]
            {
                joint.rotation.x,
                joint.rotation.y,
                joint.rotation.z,
                joint.rotation.w
            },

            euler_angles = new float[]
            {
                joint.eulerAngles.x,
                joint.eulerAngles.y,
                joint.eulerAngles.z
            },

            unity_time = joint.unityTime,
            unity_frame = joint.unityFrame
        };
    }

    private long GetUnixTimeNanoseconds()
    {
        // Approximate wall-clock Unix timestamp in nanoseconds.
        // Good enough for logging/debugging.
        // The FastAPI server can still use its own receive time for alignment.
        long unixMilliseconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        return unixMilliseconds * 1_000_000L;
    }
}

[Serializable]
public class QuestJointPacket
{
    public long quest_time_ns;

    public float unity_time;
    public int frame_id;
    public string coordinate_frame;

    public QuestUpperBodyJoints joints;
}

[Serializable]
public class QuestUpperBodyJoints
{
    // Actual headset pose.
    public QuestJointData hmd_center_eye;

    public QuestJointData root;
    public QuestJointData hips;

    public QuestJointData spine_lower;
    public QuestJointData spine_middle;
    public QuestJointData spine_upper;
    public QuestJointData chest;
    public QuestJointData neck;
    public QuestJointData head;

    public QuestJointData left_shoulder;
    public QuestJointData right_shoulder;

    public QuestJointData left_scapula;
    public QuestJointData right_scapula;

    public QuestJointData left_upper_arm;
    public QuestJointData right_upper_arm;

    public QuestJointData left_forearm;
    public QuestJointData right_forearm;

    public QuestJointData left_hand;
    public QuestJointData right_hand;

    public QuestJointData left_wrist_twist;
    public QuestJointData right_wrist_twist;

    public QuestJointData left_palm;
    public QuestJointData right_palm;
}

[Serializable]
public class QuestJointData
{
    public string joint_name;
    public string bone_id;

    public bool valid;

    public float[] position;      // [x, y, z]
    public float[] rotation;      // quaternion [x, y, z, w]
    public float[] euler_angles;  // Unity Euler angles [x, y, z]

    public float unity_time;
    public int unity_frame;
}