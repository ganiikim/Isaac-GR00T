# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


rb5_dg5f_config = {
    # ============================================================
    # Video
    # ============================================================
    # Current frame from the two cameras.
    # These names must match meta/modality.json:
    #   video.front
    #   video.second
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "front",
            "second",
        ],
    ),

    # ============================================================
    # State
    # ============================================================
    # Current proprioceptive state:
    #   rb5_arm    : 6 joint positions
    #   right_hand : 20 DG5F joint positions
    #
    # Total state dimension = 26
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "rb5_arm",
            "right_hand",
        ],
    ),

    # ============================================================
    # Action
    # ============================================================
    # Predict 16 future control steps.
    #
    # rb5_arm:
    #   joint-space action
    #   relative to current arm state
    #
    # right_hand:
    #   20-DoF joint target
    #   absolute target representation
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "rb5_arm",
            "right_hand",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # ============================================================
    # Language
    # ============================================================
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.task_description"
        ],
    ),
}


register_modality_config(
    rb5_dg5f_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
