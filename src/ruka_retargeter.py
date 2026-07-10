"""RUKA Hand Retargeting

Variant of 09_hand_retargeting.py for the RUKA-v2 hand assets copied to
`examples/retarget_helpers/hand/ruka`.
"""

import logging
import time
from pathlib import Path
from typing import Tuple, TypedDict

import zmq
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf
from zmq_utils import ZMQSubscriber, ZMQPublisher

from retarget_helpers._utils import create_conn_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ruka_retargeter")

# Makes JAX log a line every time it actually compiles a jitted function
# (as opposed to hitting the compilation cache), including the argument
# shapes/dtypes that triggered it. If `solve_retargeting` is recompiling on
# every frame instead of once, this is what will show it.
jax.config.update("jax_log_compiles", True)

subscriber = ZMQSubscriber(host="0.0.0.0", port=5052, topic="ruka_r_keypoints")
publisher = ZMQPublisher(host="0.0.0.0", port=5053)

MANO_TO_RUKA_MAPPING = {
    # Wrist / palm.
    0: "backhand",
    # Thumb.
    1: "thumb___joint_1",
    2: "thumb___joint_2",
    3: "thumb___joint_3",
    4: "thumb_actual_tip",
    # Index.
    5: "mcp",
    6: "pip",
    7: "finger___joint_3",
    8: "index_actual_tip",
    # Middle.
    9: "mcp_2",
    10: "pip_2",
    11: "finger___joint_3_2",
    12: "middle_actual_tip",
    # Ring.
    13: "mcp_3",
    14: "pip_3",
    15: "finger___joint_3_3",
    16: "ring_actual_tip",
    # Little.
    17: "mcp_4",
    18: "pinky___joint_2",
    19: "pinky___joint_3",
    20: "pinky_actual_tip",
}


class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching relative keypoint/link vectors."""
    global_alignment: float
    """Global alignment weight, by matching keypoint positions to robot links."""


def load_ruka_urdf(robot_urdf_path: Path) -> yourdfpy.URDF:
    """Load RUKA URDF while resolving its package://assets mesh paths."""
    base_path = robot_urdf_path.parent

    def filename_handler(fname: str) -> str:
        if fname.startswith("package://assets/"):
            fname = fname.removeprefix("package://assets/")
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def get_mapping_from_mano_to_ruka(robot: pk.Robot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get corresponding RUKA link indices and MANO joint indices."""
    ruka_link_indices = []
    mano_joint_indices = []
    for mano_idx, link_name in MANO_TO_RUKA_MAPPING.items():
        if link_name not in robot.links.names:
            raise ValueError(f"RUKA link '{link_name}' is not present in the URDF.")
        ruka_link_indices.append(robot.links.names.index(link_name))
        mano_joint_indices.append(mano_idx)

    return jnp.array(ruka_link_indices), jnp.array(mano_joint_indices)

@jdc.jit
def calibrate_scale(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    ruka_link_retarget_indices: jnp.ndarray,
    mano_joint_retarget_indices: jnp.ndarray,
    mano_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Batch solve over N frames to learn the MANO→RUKA morphology scale matrix.

    Runs once at startup. Returns a (n_retarget, n_retarget) matrix where
    entry [i,j] is the ratio of MANO inter-keypoint distance to RUKA
    inter-link distance for that pair. Averaged across all calibration frames
    for a robust estimate. Fixed as a constant in every subsequent fast solve.
    """
    n_retarget = len(mano_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    class ManoJointsScaleVar(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_T_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_scale = ManoJointsScaleVar(jnp.arange(timesteps))

    @jaxls.Cost.factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_scale: ManoJointsScaleVar,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_T_world_root]
        T_world_link = T_world_root @ T_root_link

        mano_pos = keypoints[mano_joint_retarget_indices]
        robot_pos = T_world_link.translation()[ruka_link_retarget_indices]

        delta_mano = mano_pos[:, None] - mano_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        scale = var_values[var_scale][..., None]
        residual_position_delta = (
            (delta_mano - delta_robot * scale)
            * (1 - jnp.eye(n_retarget)[..., None])
            * mano_mask[..., None]
        )

        dm_norm = delta_mano / jnp.linalg.norm(delta_mano + 1e-6, axis=-1, keepdims=True)
        dr_norm = delta_robot / jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True)
        residual_angle_delta = (
            (1 - (dm_norm * dr_norm).sum(axis=-1))
            * (1 - jnp.eye(n_retarget))
            * mano_mask
        )

        return jnp.concatenate(
            [residual_position_delta.flatten(), residual_angle_delta.flatten()]
        ) * 10.0

    @jaxls.Cost.factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        T_world_root = var_values[var_T_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[ruka_link_retarget_indices]
        keypoint_pos = keypoints[mano_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * 1.0

    solution = (
        jaxls.LeastSquaresProblem(
            costs=[
                retargeting_cost(var_T_world_root, var_joints, var_scale, target_keypoints),
                pc_alignment_cost(var_T_world_root, var_joints, target_keypoints),
                pk.costs.limit_constraint(
                    jax.tree.map(lambda x: x[None], robot), var_joints
                ),
            ],
            variables=[var_joints, var_T_world_root, var_scale],
        )
        .analyze()
        .solve(verbose=False)
    )
    return solution[var_scale].mean(axis=0)


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    ruka_link_retarget_indices: jnp.ndarray,
    mano_joint_retarget_indices: jnp.ndarray,
    mano_mask: jnp.ndarray,
    scale: jnp.ndarray,
    weights: RetargetingWeights,
    prev_joints: jnp.ndarray,
    prev_T_world_root: jaxlie.SE3,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve single-frame retargeting with warm start."""
    var_joints = robot.joint_var_cls(0)
    var_T_world_root = jaxls.SE3Var(0)

    @jaxls.Cost.factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_T_world_root]
        T_world_link = T_world_root @ T_root_link

        mano_pos = keypoints[mano_joint_retarget_indices]
        robot_pos = T_world_link.translation()[ruka_link_retarget_indices]

        delta_mano = mano_pos[:, None] - mano_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        residual_position_delta = (
            (delta_mano - delta_robot * scale[..., None])
            * (1 - jnp.eye(delta_mano.shape[0])[..., None])
            * mano_mask[..., None]
        )

        delta_mano_normalized = delta_mano / jnp.linalg.norm(
            delta_mano + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = (
            (1 - (delta_mano_normalized * delta_robot_normalized).sum(axis=-1))
            * (1 - jnp.eye(delta_mano.shape[0]))
            * mano_mask
        )

        return (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()],
                axis=0,
            )
            * weights["local_alignment"]
        )

    @jaxls.Cost.factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        T_world_root = var_values[var_T_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[ruka_link_retarget_indices]
        keypoint_pos = keypoints[mano_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    costs = [
        retargeting_cost(var_T_world_root, var_joints, target_keypoints),
        pc_alignment_cost(var_T_world_root, var_joints, target_keypoints),
        pk.costs.limit_constraint(robot, var_joints),
    ]

    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[var_joints, var_T_world_root])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            initial_vals=jaxls.VarValues.make(
                [var_joints.with_value(prev_joints), var_T_world_root.with_value(prev_T_world_root)]
            ),
            termination=jaxls.TerminationConfig(max_iterations=10),
        )
    )
    return solution[var_T_world_root], solution[var_joints]


def main():
    asset_dir = Path(__file__).parent / "retarget_helpers" / "hand"
    robot_urdf_path = asset_dir / "ruka" / "robot.urdf"

    try:
        urdf = load_ruka_urdf(robot_urdf_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Expected RUKA assets at `examples/retarget_helpers/hand/ruka`."
        ) from exc

    default_joint_cfg = jnp.zeros(len(urdf.actuated_joints))
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_joint_cfg)

    ruka_link_idx, mano_joint_idx = get_mapping_from_mano_to_ruka(robot)
    mano_mask = create_conn_tree(robot, ruka_link_idx)

    # Thumb diagnostics — indices into the *actuated joint* config vector
    # (`joints_np` below), used to log whether the thumb is actually reaching
    # its mechanical limits or stalling well short of them.
    thumb_joint_names = ["thumb_cmc", "thumb_mcp", "thumb_ip"]
    thumb_actuated_idx = [
        robot.joints.actuated_names.index(name) for name in thumb_joint_names
    ]
    for name, idx in zip(thumb_joint_names, thumb_actuated_idx):
        lower = float(robot.joints.lower_limits[idx])
        upper = float(robot.joints.upper_limits[idx])
        logger.info(f"joint limit: {name} (idx {idx}) = [{lower:.3f}, {upper:.3f}] rad")

    # Index into the *retargeted MANO/RUKA* pair ordering (0=wrist,
    # 1..4=thumb joint_1/2/3/tip) — used to inspect the calibrated scale
    # submatrix for the thumb chain specifically.
    thumb_retarget_idx = [0, 1, 2, 3, 4]

    # Viser visualizer.
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=True, axes_length=0.05)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(onp.zeros(len(urdf.actuated_joints)))

    with server.gui.add_folder("Retargeting Weights"):
        gui_local = server.gui.add_slider(
            "Local alignment", min=0.1, max=50.0, step=0.1, initial_value=5.0
        )
        gui_global = server.gui.add_slider(
            "Global alignment", min=0.1, max=50.0, step=0.1, initial_value=5.0
        )

    # --- Calibration phase ---
    CALIB_FRAMES = 30
    calib_buf = []
    print("===== RUKA RETARGETING PROCESS =====")
    print(
        f"== Calibrating scale ({CALIB_FRAMES} frames) — move your hand around, "
        "including opposing your thumb across your palm =="
    )
    while len(calib_buf) < CALIB_FRAMES:
        kp = subscriber.recv(flags=zmq.NOBLOCK)
        if kp is not None:
            calib_buf.append(kp)
            print(f"\r  [{len(calib_buf)}/{CALIB_FRAMES}]", end="", flush=True)
    print("\n== Solving for MANO→RUKA scale... ==")
    t0 = time.perf_counter()
    scale = calibrate_scale(
        robot=robot,
        target_keypoints=jnp.array(jnp.stack(calib_buf)),
        ruka_link_retarget_indices=ruka_link_idx,
        mano_joint_retarget_indices=mano_joint_idx,
        mano_mask=mano_mask,
    )
    scale.block_until_ready()
    logger.info(f"calibrate_scale: compile + solve took {time.perf_counter() - t0:.2f}s")
    thumb_scale_submatrix = onp.array(scale)[onp.ix_(thumb_retarget_idx, thumb_retarget_idx)]
    logger.info(
        "calibrated scale, wrist/thumb chain (rows/cols = wrist, thumb_1, thumb_2, "
        f"thumb_3, thumb_tip):\n{thumb_scale_submatrix}"
    )
    print("== Calibration done — entering fast mode ==")

    # Warm-start state — updated each frame so LM starts close to the solution.
    prev_joints = jnp.zeros(robot.joints.num_actuated_joints)
    prev_T_world_root = jaxlie.SE3.identity()

    frame_idx = 0
    while True:
        keypoints = subscriber.recv(flags=zmq.NOBLOCK)
        if keypoints is not None:
            keypoints = jnp.array(keypoints)
            if frame_idx == 0:
                logger.info(
                    f"first frame: keypoints shape={keypoints.shape} dtype={keypoints.dtype}"
                )

            weights = RetargetingWeights(
                local_alignment=jnp.array(gui_local.value),
                global_alignment=jnp.array(gui_global.value),
            )

            t0 = time.perf_counter()
            T_world_root, joints = solve_retargeting(
                robot=robot,
                target_keypoints=keypoints,
                ruka_link_retarget_indices=ruka_link_idx,
                mano_joint_retarget_indices=mano_joint_idx,
                mano_mask=mano_mask,
                scale=scale,
                weights=weights,
                prev_joints=prev_joints,
                prev_T_world_root=prev_T_world_root,
            )
            joints.block_until_ready()
            elapsed = time.perf_counter() - t0
            # First frame is expected to be slow (JIT compile). Anything else
            # slow afterwards means it's recompiling instead of hitting cache.
            if frame_idx == 0:
                logger.info(f"frame {frame_idx}: solve_retargeting compile + solve took {elapsed:.2f}s")
            elif elapsed > 0.05:
                logger.warning(
                    f"frame {frame_idx}: solve_retargeting took {elapsed * 1000:.1f}ms (unexpectedly slow — possible recompile)"
                )
            frame_idx += 1

            prev_joints = joints
            prev_T_world_root = T_world_root

            joints_np = onp.array(joints)
            root_wxyz_xyz = onp.array(T_world_root.wxyz_xyz)
            base_frame.wxyz = root_wxyz_xyz[:4]
            base_frame.position = root_wxyz_xyz[4:]
            urdf_vis.update_cfg(joints_np)

            if frame_idx % 30 == 0:
                thumb_state = ", ".join(
                    f"{name}={joints_np[idx]:+.3f} (limits [{float(robot.joints.lower_limits[idx]):+.2f}, {float(robot.joints.upper_limits[idx]):+.2f}])"
                    for name, idx in zip(thumb_joint_names, thumb_actuated_idx)
                )
                logger.info(f"frame {frame_idx}: {thumb_state}")

            server.scene.add_point_cloud(
                "/keypoints",
                onp.array(keypoints).reshape(-1, 3),
                colors=onp.full((21, 3), (0, 100, 255), dtype=onp.uint8),
                point_size=0.004,
            )

            print(joints_np)
            publisher.pub(data_array=joints_np, topic_name="ruka_r_joints")

if __name__ == "__main__":
    main()
