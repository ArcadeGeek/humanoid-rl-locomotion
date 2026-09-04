"""PPO configuration for v4-N native left/right symmetry augmentation."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ..mdp.v4_symmetry import compute_g1_left_right_symmetric_states
from .v4b_ppo_cfg import V4BDirectYawPPORunnerCfg


@configclass
class V4NSymmetryAugmentationPPORunnerCfg(V4BDirectYawPPORunnerCfg):
    """v4-B unchanged, plus native RSL-RL left/right data augmentation only."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_g1_left_right_symmetric_states,
            mirror_loss_coeff=0.0,
        )
