from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Robot Motion History Server",
    version="1.0.0",
)


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class RobotMotionSample(BaseModel):
    timestamp: float
    time_from_window_start: float

    position: Vector3
    heading: Vector3


class RobotMotionHistoryWindow(BaseModel):
    schema_version: str

    window_id: str
    source_id: str

    sent_at_unix_seconds: float

    window_duration_seconds: float
    sample_count: int = Field(ge=1)

    samples: List[RobotMotionSample]


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


@app.post("/robot_motion_history")
async def receive_robot_motion_history(
    window: RobotMotionHistoryWindow,
):
    if window.sample_count != len(window.samples):
        raise HTTPException(
            status_code=400,
            detail=(
                "sample_count does not match the number "
                "of samples in the payload."
            ),
        )

    print()
    print("Received robot motion-history window")
    print("------------------------------------")
    print(f"Window ID: {window.window_id}")
    print(f"Source: {window.source_id}")
    print(f"Schema: {window.schema_version}")
    print(
        f"Duration: "
        f"{window.window_duration_seconds:.3f} seconds"
    )
    print(f"Samples: {window.sample_count}")

    if window.samples:
        first = window.samples[0]
        last = window.samples[-1]

        print(
            "Start position: "
            f"({first.position.x:.3f}, "
            f"{first.position.y:.3f}, "
            f"{first.position.z:.3f})"
        )

        print(
            "End position: "
            f"({last.position.x:.3f}, "
            f"{last.position.y:.3f}, "
            f"{last.position.z:.3f})"
        )

        print(
            "Latest heading: "
            f"({last.heading.x:.3f}, "
            f"{last.heading.y:.3f}, "
            f"{last.heading.z:.3f})"
        )

    print("------------------------------------")
    print()

    #
    # This is where the eventual trajectory-prediction model,
    # database insertion, message queue, etc. can be called.
    #

    return {
        "accepted": True,
        "window_id": window.window_id,
        "samples_received": len(window.samples),
    }