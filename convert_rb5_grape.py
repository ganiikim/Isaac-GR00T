import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

HF_USERNAME = "ganikim"

FPS = 30.0

TASK = (
    "Sort one bunch of grapes by color: "
    "place green grapes on the right and red grapes on the left."
)

STATE_DIM = 26
ACTION_DIM = 26

EMBODIMENT_TAG = "NEW_EMBODIMENT"

MODALITY_CONFIG_RELATIVE_PATH = (
    Path("examples")
    / "RB5_DG5F"
    / "rb5_dg5f_config.py"
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Convert RB5 + DG5F teleoperation data "
            "to LeRobot v2.1, generate GR00T statistics, "
            "and upload the dataset to Hugging Face."
        )
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help=(
            "Source dataset folder ID. "
            "Example: --dataset-id 1 "
            "-> ~/rb5_teleop/dataset/1"
        ),
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help=(
            "Output/Hugging Face dataset name. "
            "Example: rb5_grape_sort_v2"
        ),
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def write_json(path, obj):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=4,
            ensure_ascii=False,
        )


def write_jsonl(path, rows):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_video_info(path):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {path}"
        )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    cap.release()

    return {
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }


# ============================================================
# Episode discovery
# ============================================================

def discover_episodes(source_root):

    if not source_root.exists():

        raise FileNotFoundError(
            "Source dataset directory does not exist:\n"
            f"{source_root}"
        )

    if not source_root.is_dir():

        raise RuntimeError(
            "Source path is not a directory:\n"
            f"{source_root}"
        )

    episodes = []

    print()
    print("=" * 70)
    print("SEARCHING EPISODES")
    print("=" * 70)

    for episode_dir in source_root.iterdir():

        if not episode_dir.is_dir():
            continue

        required_files = [
            episode_dir / "episode.json",
            episode_dir / "front.mp4",
            episode_dir / "second.mp4",
        ]

        missing_files = [
            path.name
            for path in required_files
            if not path.is_file()
        ]

        if not missing_files:

            episodes.append(
                episode_dir
            )

            print(
                f"[VALID]   {episode_dir.name}"
            )

        else:

            print(
                f"[SKIPPED] {episode_dir.name} "
                f"missing={missing_files}"
            )

    # --------------------------------------------------------
    # Numeric directory sorting
    #
    # 1, 2, 3, 10
    #
    # instead of
    #
    # 1, 10, 2, 3
    # --------------------------------------------------------

    def sort_key(path):

        try:

            return (
                0,
                int(path.name),
            )

        except ValueError:

            return (
                1,
                path.name,
            )

    episodes.sort(
        key=sort_key
    )

    if not episodes:

        raise RuntimeError(
            "No valid episodes found in:\n"
            f"{source_root}"
        )

    print()
    print(
        f"Found {len(episodes)} "
        "valid episodes."
    )

    print(
        "Episodes:",
        [
            path.name
            for path in episodes
        ],
    )

    return episodes


# ============================================================
# GR00T statistics generation
# ============================================================

def generate_stats(
    repo_root,
    output_root,
):

    print()
    print("=" * 70)
    print("GENERATING GR00T DATASET STATISTICS")
    print("=" * 70)

    stats_script = (
        repo_root
        / "gr00t"
        / "data"
        / "stats.py"
    )

    modality_config = (
        repo_root
        / MODALITY_CONFIG_RELATIVE_PATH
    )

    if not stats_script.is_file():

        raise FileNotFoundError(
            "GR00T stats script not found:\n"
            f"{stats_script}"
        )

    if not modality_config.is_file():

        raise FileNotFoundError(
            "Modality config not found:\n"
            f"{modality_config}"
        )

    print(
        f"Dataset         : {output_root}"
    )

    print(
        f"Embodiment      : {EMBODIMENT_TAG}"
    )

    print(
        f"Modality config : {modality_config}"
    )

    # --------------------------------------------------------
    # Use the same Python interpreter that is running
    # convert_rb5_grape.py.
    #
    # This is safer than hardcoding "python".
    # --------------------------------------------------------

    command = [
        sys.executable,
        str(stats_script),

        "--dataset-path",
        str(output_root),

        "--embodiment-tag",
        EMBODIMENT_TAG,

        "--modality-config-path",
        str(modality_config),
    ]

    print()
    print("Running:")
    print(
        " ".join(
            str(x)
            for x in command
        )
    )
    print()

    subprocess.run(
        command,
        cwd=str(repo_root),
        check=True,
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    stats_path = (
        output_root
        / "meta"
        / "stats.json"
    )

    relative_stats_path = (
        output_root
        / "meta"
        / "relative_stats.json"
    )

    if not stats_path.is_file():

        raise RuntimeError(
            "GR00T statistics generation finished, "
            "but stats.json was not generated:\n"
            f"{stats_path}"
        )

    if not relative_stats_path.is_file():

        raise RuntimeError(
            "GR00T statistics generation finished, "
            "but relative_stats.json was not generated:\n"
            f"{relative_stats_path}"
        )

    print()
    print("=" * 70)
    print("STATISTICS GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"stats.json          : "
        f"{stats_path}"
    )

    print(
        f"relative_stats.json : "
        f"{relative_stats_path}"
    )


# ============================================================
# Hugging Face upload
# ============================================================

def upload_to_huggingface(
    output_root,
    dataset_name,
):

    repo_id = (
        f"{HF_USERNAME}/"
        f"{dataset_name}"
    )

    print()
    print("=" * 70)
    print("HUGGING FACE UPLOAD")
    print("=" * 70)

    print(
        f"Repository : {repo_id}"
    )

    print(
        f"Local path : {output_root}"
    )

    # --------------------------------------------------------
    # Make sure hf CLI exists
    # --------------------------------------------------------

    hf_command = shutil.which(
        "hf"
    )

    if hf_command is None:

        raise RuntimeError(
            "'hf' command was not found.\n"
            "Install/login to Hugging Face CLI first."
        )

    # --------------------------------------------------------
    # Check login
    # --------------------------------------------------------

    print()
    print("Checking Hugging Face login...")

    subprocess.run(
        [
            hf_command,
            "auth",
            "whoami",
        ],
        check=True,
    )

    # --------------------------------------------------------
    # Create dataset repository
    #
    # If it already exists, keep using it.
    # --------------------------------------------------------

    print()
    print(
        "Creating/checking "
        "Hugging Face dataset repository..."
    )

    subprocess.run(
        [
            hf_command,
            "repo",
            "create",
            repo_id,

            "--repo-type",
            "dataset",

            "--exist-ok",
        ],
        check=True,
    )

    # --------------------------------------------------------
    # Upload complete dataset directory
    # --------------------------------------------------------

    print()
    print(
        "Uploading dataset..."
    )

    subprocess.run(
        [
            hf_command,
            "upload",
            repo_id,

            str(output_root),

            ".",

            "--repo-type",
            "dataset",
        ],
        check=True,
    )

    print()
    print("=" * 70)
    print("HUGGING FACE UPLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Dataset : {repo_id}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Isaac-GR00T repository root
    #
    # convert_rb5_grape.py is expected to be:
    #
    # ~/Isaac-GR00T/convert_rb5_grape.py
    # --------------------------------------------------------

    repo_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    # --------------------------------------------------------
    # Source
    #
    # --dataset-id 1
    #
    # ->
    #
    # ~/rb5_teleop/dataset/1
    # --------------------------------------------------------

    source_root = (
        Path.home()
        / "rb5_teleop"
        / "dataset"
        / str(args.dataset_id)
    )

    # --------------------------------------------------------
    # Output
    #
    # --dataset-name rb5_grape_sort_v2
    #
    # ->
    #
    # ~/Isaac-GR00T/demo_data/rb5_grape_sort_v2
    # --------------------------------------------------------

    output_root = (
        repo_root
        / "demo_data"
        / args.dataset_name
    )

    # --------------------------------------------------------
    # Discover episodes automatically
    # --------------------------------------------------------

    episode_dirs = discover_episodes(
        source_root
    )

    print()
    print("=" * 70)
    print(
        "RB5 + DG5F -> "
        "GR00T N1.7 DATASET CONVERTER"
    )
    print("=" * 70)

    print(
        f"Dataset ID   : "
        f"{args.dataset_id}"
    )

    print(
        f"Dataset name : "
        f"{args.dataset_name}"
    )

    print(
        f"Source       : "
        f"{source_root}"
    )

    print(
        f"Output       : "
        f"{output_root}"
    )

    print(
        f"Episodes     : "
        f"{len(episode_dirs)}"
    )

    # ========================================================
    # Clean existing output
    # ========================================================

    if output_root.exists():

        print()
        print(
            "Removing existing dataset:"
        )

        print(
            f"  {output_root}"
        )

        shutil.rmtree(
            output_root
        )

    # ========================================================
    # Output directories
    # ========================================================

    meta_dir = (
        output_root
        / "meta"
    )

    data_dir = (
        output_root
        / "data"
        / "chunk-000"
    )

    front_dir = (
        output_root
        / "videos"
        / "chunk-000"
        / "observation.images.front"
    )

    second_dir = (
        output_root
        / "videos"
        / "chunk-000"
        / "observation.images.second"
    )

    for path in [
        meta_dir,
        data_dir,
        front_dir,
        second_dir,
    ]:

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Dataset-level accumulators
    # ========================================================

    episodes_meta = []

    global_index = 0

    total_frames = 0

    # ========================================================
    # Convert episodes
    # ========================================================

    for (
        episode_index,
        source_dir,
    ) in enumerate(
        episode_dirs
    ):

        source_id = (
            source_dir.name
        )

        print()
        print(
            f"[{episode_index + 1}/"
            f"{len(episode_dirs)}] "
            f"source episode "
            f"{source_id}"
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

        # ----------------------------------------------------
        # Load original JSON
        # ----------------------------------------------------

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            src = json.load(f)

        if "data" not in src:

            raise RuntimeError(
                "'data' key missing in:\n"
                f"{json_path}"
            )

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
        # Validate lengths
        # ----------------------------------------------------

        arrays = {

            "timestamp":
                timestamp_ns,

            "rb5_state":
                rb5_state,

            "rb5_action":
                rb5_action,

            "hand_state":
                hand_state,

            "hand_action":
                hand_action,
        }

        lengths = {

            key: len(value)

            for key, value
            in arrays.items()
        }

        if (
            len(
                set(
                    lengths.values()
                )
            )
            != 1
        ):

            raise RuntimeError(
                "Length mismatch in "
                f"episode {source_id}: "
                f"{lengths}"
            )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if (
            rb5_state.ndim != 2
            or
            rb5_state.shape[1] != 6
        ):

            raise RuntimeError(
                "RB5 state dimension error "
                f"in episode {source_id}: "
                f"{rb5_state.shape}"
            )

        if (
            rb5_action.ndim != 2
            or
            rb5_action.shape[1] != 6
        ):

            raise RuntimeError(
                "RB5 action dimension error "
                f"in episode {source_id}: "
                f"{rb5_action.shape}"
            )

        if (
            hand_state.ndim != 2
            or
            hand_state.shape[1] != 20
        ):

            raise RuntimeError(
                "Hand state dimension error "
                f"in episode {source_id}: "
                f"{hand_state.shape}"
            )

        if (
            hand_action.ndim != 2
            or
            hand_action.shape[1] != 20
        ):

            raise RuntimeError(
                "Hand action dimension error "
                f"in episode {source_id}: "
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

        json_frames = (
            len(timestamp_ns)
        )

        print(
            f"  JSON   : "
            f"{json_frames}"
        )

        print(
            f"  Front  : "
            f"{front_info['frames']} "
            "frames"
        )

        print(
            f"  Second : "
            f"{second_info['frames']} "
            "frames"
        )

        # ----------------------------------------------------
        # Use only frames available in ALL modalities
        # ----------------------------------------------------

        episode_length = min(
            json_frames,
            front_info["frames"],
            second_info["frames"],
        )

        if episode_length <= 0:

            raise RuntimeError(
                f"Episode {source_id} "
                "has no usable frames."
            )

        if (
            episode_length
            != json_frames
            or
            episode_length
            != front_info["frames"]
            or
            episode_length
            != second_info["frames"]
        ):

            print(
                "  WARNING: frame count mismatch."
            )

            print(
                f"  Using first "
                f"{episode_length} frames."
            )

        # ----------------------------------------------------
        # Construct state
        #
        # [0:6]  = RB5
        # [6:26] = DG5F
        # ----------------------------------------------------

        state = np.concatenate(
            [
                rb5_state[
                    :episode_length
                ],

                hand_state[
                    :episode_length
                ],
            ],
            axis=1,
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Construct action
        #
        # [0:6]  = RB5
        # [6:26] = DG5F
        # ----------------------------------------------------

        action = np.concatenate(
            [
                rb5_action[
                    :episode_length
                ],

                hand_action[
                    :episode_length
                ],
            ],
            axis=1,
        ).astype(
            np.float32
        )

        if (
            state.shape
            !=
            (
                episode_length,
                STATE_DIM,
            )
        ):

            raise RuntimeError(
                "Final state shape error: "
                f"{state.shape}"
            )

        if (
            action.shape
            !=
            (
                episode_length,
                ACTION_DIM,
            )
        ):

            raise RuntimeError(
                "Final action shape error: "
                f"{action.shape}"
            )

        # ----------------------------------------------------
        # Relative timestamp
        # ----------------------------------------------------

        ts = (
            timestamp_ns[
                :episode_length
            ]
        )

        timestamp = (
            (ts - ts[0])
            / 1e9
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Build parquet rows
        # ----------------------------------------------------

        rows = []

        for frame_index in range(
            episode_length
        ):

            is_last = (
                frame_index
                ==
                episode_length - 1
            )

            row = {

                "observation.state":
                    state[
                        frame_index
                    ].tolist(),

                "action":
                    action[
                        frame_index
                    ].tolist(),

                "timestamp":
                    float(
                        timestamp[
                            frame_index
                        ]
                    ),

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
                    bool(
                        is_last
                    ),
            }

            rows.append(
                row
            )

            global_index += 1

        # ----------------------------------------------------
        # Write parquet
        # ----------------------------------------------------

        parquet_name = (
            f"episode_"
            f"{episode_index:06d}"
            f".parquet"
        )

        parquet_path = (
            data_dir
            / parquet_name
        )

        df = pd.DataFrame(
            rows
        )

        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            index=False,
        )

        # ----------------------------------------------------
        # Copy videos
        # ----------------------------------------------------

        video_name = (
            f"episode_"
            f"{episode_index:06d}"
            f".mp4"
        )

        shutil.copy2(
            front_src,
            front_dir
            / video_name,
        )

        shutil.copy2(
            second_src,
            second_dir
            / video_name,
        )

        # ----------------------------------------------------
        # Episode metadata
        # ----------------------------------------------------

        episodes_meta.append(
            {
                "episode_index":
                    episode_index,

                "tasks": [
                    TASK
                ],

                "length":
                    episode_length,
            }
        )

        total_frames += (
            episode_length
        )

        print(
            f"  -> episode_"
            f"{episode_index:06d}"
            f" : {episode_length} "
            "frames"
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
        meta_dir
        / "tasks.jsonl",
        tasks,
    )

    # ========================================================
    # episodes.jsonl
    # ========================================================

    write_jsonl(
        meta_dir
        / "episodes.jsonl",
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
        meta_dir
        / "modality.json",
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

        "codebase_version":
            "v2.1",

        "robot_type":
            "rb5_dg5f",

        "total_episodes":
            len(
                episode_dirs
            ),

        "total_frames":
            total_frames,

        "total_tasks":
            1,

        "chunks_size":
            1000,

        "fps":
            FPS,

        "splits": {

            "train":
                f"0:"
                f"{len(episode_dirs)}"
        },

        "data_path":
            "data/"
            "chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet",

        "video_path":
            "videos/"
            "chunk-{episode_chunk:03d}/"
            "{video_key}/"
            "episode_{episode_index:06d}.mp4",

        "features": {

            "action": {

                "dtype":
                    "float32",

                "shape":
                    [26],

                "names":
                    joint_names,
            },

            "observation.state": {

                "dtype":
                    "float32",

                "shape":
                    [26],

                "names":
                    joint_names,
            },

            "observation.images.front": {

                "dtype":
                    "video",

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

                    "video.height":
                        480,

                    "video.width":
                        640,

                    "video.codec":
                        "mpeg4",

                    "video.pix_fmt":
                        "yuv420p",

                    "video.is_depth_map":
                        False,

                    "video.fps":
                        FPS,

                    "video.channels":
                        3,

                    "has_audio":
                        False,
                },
            },

            "observation.images.second": {

                "dtype":
                    "video",

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

                    "video.height":
                        480,

                    "video.width":
                        640,

                    "video.codec":
                        "mpeg4",

                    "video.pix_fmt":
                        "yuv420p",

                    "video.is_depth_map":
                        False,

                    "video.fps":
                        FPS,

                    "video.channels":
                        3,

                    "has_audio":
                        False,
                },
            },

            "timestamp": {

                "dtype":
                    "float32",

                "shape":
                    [1],

                "names":
                    None,
            },

            "frame_index": {

                "dtype":
                    "int64",

                "shape":
                    [1],

                "names":
                    None,
            },

            "episode_index": {

                "dtype":
                    "int64",

                "shape":
                    [1],

                "names":
                    None,
            },

            "index": {

                "dtype":
                    "int64",

                "shape":
                    [1],

                "names":
                    None,
            },

            "task_index": {

                "dtype":
                    "int64",

                "shape":
                    [1],

                "names":
                    None,
            },

            "annotation.human.task_description": {

                "dtype":
                    "int64",

                "shape":
                    [1],

                "names":
                    None,
            },

            "next.reward": {

                "dtype":
                    "float32",

                "shape":
                    [1],

                "names":
                    None,
            },

            "next.done": {

                "dtype":
                    "bool",

                "shape":
                    [1],

                "names":
                    None,
            },
        },

        "total_chunks":
            1,

        "total_videos":
            len(
                episode_dirs
            )
            * 2,
    }

    write_json(
        meta_dir
        / "info.json",
        info,
    )

    # ========================================================
    # Conversion complete
    # ========================================================

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Output       : "
        f"{output_root}"
    )

    print(
        f"Episodes     : "
        f"{len(episode_dirs)}"
    )

    print(
        f"Total frames : "
        f"{total_frames}"
    )

    print(
        f"State dim    : "
        f"{STATE_DIM}"
    )

    print(
        f"Action dim   : "
        f"{ACTION_DIM}"
    )

    # ========================================================
    # Generate stats.json + relative_stats.json
    # ========================================================

    generate_stats(
        repo_root,
        output_root,
    )

    # ========================================================
    # Upload to Hugging Face
    # ========================================================

    upload_to_huggingface(
        output_root,
        args.dataset_name,
    )

    # ========================================================
    # All done
    # ========================================================

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)

    print(
        f"Local dataset : "
        f"{output_root}"
    )

    print(
        f"Hugging Face  : "
        f"{HF_USERNAME}/"
        f"{args.dataset_name}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()