import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SOURCE_ROOT = Path.home() / "rb5_teleop" / "dataset"

OUTPUT_ROOT = (
    Path.home()
    / "Isaac-GR00T"
    / "demo_data"
    / "rb5_grape_sort"
)

EPISODES = [
    63, 64, 65, 66, 67, 68, 69,
    71, 72, 73, 77, 78, 79,
]

FPS = 30.0

TASK = (
    "Sort one bunch of grapes by color: "
    "place green grapes on the right and red grapes on the left."
)

STATE_DIM = 26
ACTION_DIM = 26


# ============================================================
# Utilities
# ============================================================

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )


def get_video_info(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
        )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    cap.release()

    return {
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("RB5 + DG5F -> GR00T N1.7 dataset converter")
    print("=" * 70)

    # --------------------------------------------------------
    # Clean output
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            f"Removing existing dataset:\n"
            f"  {OUTPUT_ROOT}"
        )

        shutil.rmtree(OUTPUT_ROOT)

    meta_dir = OUTPUT_ROOT / "meta"

    data_dir = (
        OUTPUT_ROOT
        / "data"
        / "chunk-000"
    )

    front_dir = (
        OUTPUT_ROOT
        / "videos"
        / "chunk-000"
        / "observation.images.front"
    )

    second_dir = (
        OUTPUT_ROOT
        / "videos"
        / "chunk-000"
        / "observation.images.second"
    )

    for p in [
        meta_dir,
        data_dir,
        front_dir,
        second_dir,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Dataset-level accumulators
    # --------------------------------------------------------

    episodes_meta = []

    global_index = 0
    total_frames = 0

    # ========================================================
    # Episodes
    # ========================================================

    for episode_index, source_id in enumerate(
        EPISODES
    ):

        source_dir = (
            SOURCE_ROOT
            / str(source_id)
        )

        print()
        print(
            f"[{episode_index + 1}/{len(EPISODES)}] "
            f"source episode {source_id}"
        )

        # ----------------------------------------------------
        # Required files
        # ----------------------------------------------------

        json_path = (
            source_dir
            / "episode.json"
        )

        front_src = (
            source_dir
            / "front.mp4"
        )

        second_src = (
            source_dir
            / "second.mp4"
        )

        for p in [
            json_path,
            front_src,
            second_src,
        ]:

            if not p.exists():
                raise FileNotFoundError(
                    f"Missing file: {p}"
                )

        # ----------------------------------------------------
        # Load original JSON
        # ----------------------------------------------------

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            src = json.load(f)

        d = src["data"]

        # ----------------------------------------------------
        # Load arrays
        # ----------------------------------------------------

        timestamp_ns = np.asarray(
            d["timestamp_ns"],
            dtype=np.int64,
        )

        rb5_state = np.asarray(
            d["rb5_joint_state"],
            dtype=np.float32,
        )

        rb5_action = np.asarray(
            d["rb5_joint_ref"],
            dtype=np.float32,
        )

        hand_state = np.asarray(
            d["right_hand_state"],
            dtype=np.float32,
        )

        hand_action = np.asarray(
            d["right_hand_ref"],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        arrays = {
            "timestamp": timestamp_ns,
            "rb5_state": rb5_state,
            "rb5_action": rb5_action,
            "hand_state": hand_state,
            "hand_action": hand_action,
        }

        lengths = {
            k: len(v)
            for k, v in arrays.items()
        }

        if len(set(lengths.values())) != 1:

            raise RuntimeError(
                f"Length mismatch in episode "
                f"{source_id}: {lengths}"
            )

        if rb5_state.shape[1] != 6:
            raise RuntimeError(
                f"RB5 state dimension error: "
                f"{rb5_state.shape}"
            )

        if rb5_action.shape[1] != 6:
            raise RuntimeError(
                f"RB5 action dimension error: "
                f"{rb5_action.shape}"
            )

        if hand_state.shape[1] != 20:
            raise RuntimeError(
                f"Hand state dimension error: "
                f"{hand_state.shape}"
            )

        if hand_action.shape[1] != 20:
            raise RuntimeError(
                f"Hand action dimension error: "
                f"{hand_action.shape}"
            )

        # ----------------------------------------------------
        # Validate videos
        # ----------------------------------------------------

        front_info = get_video_info(
            front_src
        )

        second_info = get_video_info(
            second_src
        )

        json_frames = len(timestamp_ns)

        print(
            f"  JSON   : {json_frames}"
        )

        print(
            f"  Front  : {front_info['frames']} frames"
        )

        print(
            f"  Second : {second_info['frames']} frames"
        )

        # Use only frames available in ALL modalities.
        episode_length = min(
            json_frames,
            front_info["frames"],
            second_info["frames"],
        )

        if episode_length <= 0:
            raise RuntimeError(
                f"Episode {source_id} has no frames."
            )

        if episode_length != json_frames:

            print(
                "  WARNING: frame count mismatch. "
                f"Using first {episode_length} frames."
            )

        # ----------------------------------------------------
        # Construct state/action
        # ----------------------------------------------------

        state = np.concatenate(
            [
                rb5_state[:episode_length],
                hand_state[:episode_length],
            ],
            axis=1,
        ).astype(np.float32)

        action = np.concatenate(
            [
                rb5_action[:episode_length],
                hand_action[:episode_length],
            ],
            axis=1,
        ).astype(np.float32)

        assert state.shape == (
            episode_length,
            STATE_DIM,
        )

        assert action.shape == (
            episode_length,
            ACTION_DIM,
        )

        # ----------------------------------------------------
        # Relative timestamp
        # ----------------------------------------------------

        ts = timestamp_ns[:episode_length]

        timestamp = (
            (ts - ts[0])
            / 1e9
        ).astype(np.float32)

        # ----------------------------------------------------
        # Build parquet rows
        # ----------------------------------------------------

        rows = []

        for frame_index in range(
            episode_length
        ):

            is_last = (
                frame_index
                == episode_length - 1
            )

            row = {
                "observation.state":
                    state[frame_index].tolist(),

                "action":
                    action[frame_index].tolist(),

                "timestamp":
                    float(timestamp[frame_index]),

                "annotation.human.task_description":
                    0,

                "task_index":
                    0,

                "episode_index":
                    episode_index,

                "frame_index":
                    frame_index,

                "index":
                    global_index,

                "next.reward":
                    0.0,

                "next.done":
                    bool(is_last),
            }

            rows.append(row)

            global_index += 1

        # ----------------------------------------------------
        # Write parquet
        # ----------------------------------------------------

        parquet_name = (
            f"episode_{episode_index:06d}.parquet"
        )

        parquet_path = (
            data_dir
            / parquet_name
        )

        df = pd.DataFrame(rows)

        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            index=False,
        )

        # ----------------------------------------------------
        # Copy videos
        # ----------------------------------------------------

        video_name = (
            f"episode_{episode_index:06d}.mp4"
        )

        shutil.copy2(
            front_src,
            front_dir / video_name,
        )

        shutil.copy2(
            second_src,
            second_dir / video_name,
        )

        # ----------------------------------------------------
        # Episode metadata
        # ----------------------------------------------------

        episodes_meta.append({
            "episode_index":
                episode_index,

            "tasks": [
                TASK
            ],

            "length":
                episode_length,
        })

        total_frames += episode_length

        print(
            f"  -> episode_{episode_index:06d}"
            f" : {episode_length} frames"
        )

    # ========================================================
    # tasks.jsonl
    # ========================================================

    tasks = [
        {
            "task_index": 0,
            "task": TASK,
        }
    ]

    write_jsonl(
        meta_dir / "tasks.jsonl",
        tasks,
    )

    # ========================================================
    # episodes.jsonl
    # ========================================================

    write_jsonl(
        meta_dir / "episodes.jsonl",
        episodes_meta,
    )

    # ========================================================
    # modality.json
    # ========================================================

    modality = {

        "state": {

            "rb5_arm": {
                "start": 0,
                "end": 6,
            },

            "right_hand": {
                "start": 6,
                "end": 26,
            },
        },

        "action": {

            "rb5_arm": {
                "start": 0,
                "end": 6,
            },

            "right_hand": {
                "start": 6,
                "end": 26,
            },
        },

        "video": {

            "front": {
                "original_key":
                    "observation.images.front"
            },

            "second": {
                "original_key":
                    "observation.images.second"
            },
        },

        "annotation": {

            "human.task_description": {
                "original_key":
                    "task_index"
            }
        },
    }

    write_json(
        meta_dir / "modality.json",
        modality,
    )

    # ========================================================
    # info.json
    # ========================================================

    joint_names = (
        [
            f"rb5_joint_{i + 1}"
            for i in range(6)
        ]
        +
        [
            f"dg5f_joint_{i + 1}"
            for i in range(20)
        ]
    )

    info = {

        "codebase_version": "v2.1",

        "robot_type": "rb5_dg5f",

        "total_episodes":
            len(EPISODES),

        "total_frames":
            total_frames,

        "total_tasks": 1,

        "chunks_size": 1000,

        "fps": FPS,

        "splits": {
            "train":
                f"0:{len(EPISODES)}"
        },

        "data_path":
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet",

        "video_path":
            "videos/chunk-{episode_chunk:03d}/"
            "{video_key}/"
            "episode_{episode_index:06d}.mp4",

        "features": {

            "action": {
                "dtype": "float32",
                "shape": [26],
                "names": joint_names,
            },

            "observation.state": {
                "dtype": "float32",
                "shape": [26],
                "names": joint_names,
            },

            "observation.images.front": {

                "dtype": "video",

                "shape": [
                    480,
                    640,
                    3,
                ],

                "names": [
                    "height",
                    "width",
                    "channels",
                ],

                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },

            "observation.images.second": {

                "dtype": "video",

                "shape": [
                    480,
                    640,
                    3,
                ],

                "names": [
                    "height",
                    "width",
                    "channels",
                ],

                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },

            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },

            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "annotation.human.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "next.reward": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },

            "next.done": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
        },

        "total_chunks": 1,

        "total_videos":
            len(EPISODES) * 2,
    }

    write_json(
        meta_dir / "info.json",
        info,
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Output       : {OUTPUT_ROOT}"
    )

    print(
        f"Episodes     : {len(EPISODES)}"
    )

    print(
        f"Total frames : {total_frames}"
    )

    print(
        f"State dim    : {STATE_DIM}"
    )

    print(
        f"Action dim   : {ACTION_DIM}"
    )

    print()
    print(
        "NOTE: stats.json / relative_stats.json "
        "have not been generated yet."
    )


if __name__ == "__main__":
    main()