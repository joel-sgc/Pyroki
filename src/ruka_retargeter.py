"""RUKA Hand Retargeting

Variant of 09_hand_retargeting.py for the RUKA-v2 hand assets copied to
`examples/retarget_helpers/hand/ruka`.
"""

import argparse
import pickle
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
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf
from zmq_utils import ZMQSubscriber, ZMQPublisher

from retarget_helpers._utils import create_conn_tree

subscriber = ZMQSubscriber(host="localhost", port=5052, topic="ruka_r_keypoints")
publisher = ZMQPublisher(host="localhost", port=5053)

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


def old_main():
    """Main function for RUKA hand retargeting."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Limit timesteps for faster debugging.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Only load the URDF and print mapping/joint information.",
    )
    parser.add_argument(
        "--solve-only",
        action="store_true",
        help="Solve once, print output shapes, and exit without opening the viewer.",
    )
    args = parser.parse_args()

    asset_dir = Path(__file__).parent / "retarget_helpers" / "hand"
    robot_urdf_path = asset_dir / "ruka" / "robot.urdf"

    try:
        urdf = load_ruka_urdf(robot_urdf_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Expected RUKA assets at `examples/retarget_helpers/hand/ruka`."
        ) from exc

    # A zero default is less curled than the midpoint of RUKA's joint limits.
    default_joint_cfg = jnp.zeros(len(urdf.actuated_joints))
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_joint_cfg)

    ruka_link_idx, mano_joint_idx = get_mapping_from_mano_to_ruka(robot)
    mano_mask = create_conn_tree(robot, ruka_link_idx)

    if args.probe:
        print(f"URDF: {robot_urdf_path}")
        print(f"links: {robot.links.num_links}")
        print(f"joints: {robot.joints.num_joints}")
        print(f"actuated joints: {robot.joints.num_actuated_joints}")
        print("actuated joint order:")
        for i, joint_name in enumerate(robot.joints.actuated_names):
            print(f"  {i:02d}: {joint_name}")
        print("MANO -> RUKA mapping:")
        for mano_idx, link_idx in zip(mano_joint_idx, ruka_link_idx):
            print(f"  {int(mano_idx):02d}: {robot.links.names[int(link_idx)]}")
        return

    keypoints = dexycb_motion_data["world_hand_joints"]
    assert not onp.isnan(keypoints).any()
    if args.timesteps is not None:
        keypoints = keypoints[: args.timesteps]
    num_timesteps = keypoints.shape[0]
    num_mano_joints = keypoints.shape[1]

    contact_points_per_frame = dexycb_motion_data["contact_object_points"]
    if args.timesteps is not None:
        contact_points_per_frame = contact_points_per_frame[: args.timesteps]

    object_mesh_vertices = dexycb_motion_data["object_mesh_vertices"]
    object_mesh_faces = dexycb_motion_data["object_mesh_faces"]
    object_pose_list = dexycb_motion_data["object_poses"]
    if args.timesteps is not None:
        object_pose_list = object_pose_list[: args.timesteps]
    mesh = trimesh.Trimesh(object_mesh_vertices, object_mesh_faces)

    default_weights = RetargetingWeights(
        local_alignment=10.0,
        global_alignment=1.0,
        joint_smoothness=2.0,
        root_smoothness=2.0,
    )

    if args.solve_only:
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=keypoints,
            ruka_link_retarget_indices=ruka_link_idx,
            mano_joint_retarget_indices=mano_joint_idx,
            mano_mask=mano_mask,
            weights=default_weights,
        )
        print(f"Ts_world_root: {Ts_world_root.wxyz_xyz.shape}")
        print(f"joints: {joints.shape}")
        print(f"actuated joint order: {robot.joints.actuated_names}")
        return

    server = viser.ViserServer()

    server.scene.add_frame(
        "/scene_offset",
        show_axes=False,
        position=(-0.15415953, -0.73598871, 0.93434792),
        wxyz=(-0.381870867, 0.92421569, 0.0, 2.0004992e-32),
    )
    hand_mesh = server.scene.add_mesh_simple(
        "/scene_offset/hand_mesh",
        vertices=dexycb_motion_data["world_hand_vertices"][0, :, :],
        faces=dexycb_motion_data["hand_mesh_faces"],
        opacity=0.35,
    )
    base_frame = server.scene.add_frame("/scene_offset/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/scene_offset/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    object_handle = server.scene.add_mesh_trimesh("/scene_offset/object", mesh)
    server.scene.add_grid("/grid", 2.0, 2.0)

    weights = pk.viewer.WeightTuner(
        server,
        default_weights,  # type: ignore
    )

    Ts_world_root, joints = None, None

    def generate_trajectory():
        nonlocal Ts_world_root, joints
        gen_button.disabled = True
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=keypoints,
            ruka_link_retarget_indices=ruka_link_idx,
            mano_joint_retarget_indices=mano_joint_idx,
            mano_mask=mano_mask,
            weights=weights.get_weights(),  # type: ignore
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            server.scene.add_point_cloud(
                "/scene_offset/target_keypoints",
                onp.array(keypoints[tstep]).reshape(-1, 3),
                onp.array((0, 0, 255))[None]
                .repeat(num_mano_joints, axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="sparkle",
            )
            server.scene.add_point_cloud(
                "/scene_offset/contact_points",
                onp.array(contact_points_per_frame[tstep]).reshape(-1, 3),
                onp.array((255, 0, 0))[None]
                .repeat(len(contact_points_per_frame[tstep]), axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="circle",
            )
            hand_mesh.vertices = dexycb_motion_data["world_hand_vertices"][tstep, :, :]
            object_handle.position = object_pose_list[tstep][:3, 3]
            object_handle.wxyz = R.from_matrix(object_pose_list[tstep][:3, :3]).as_quat(
                scalar_first=True
            )

        time.sleep(0.05)


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    ruka_link_retarget_indices: jnp.ndarray,
    mano_joint_retarget_indices: jnp.ndarray,
    mano_mask: jnp.ndarray,
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
            (delta_mano - delta_robot)
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

    default_weights = RetargetingWeights(
        local_alignment=10.0,
        global_alignment=1.0,
    )

    # Viser visualizer.
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=True, axes_length=0.05)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(onp.zeros(len(urdf.actuated_joints)))

    # Warm-start state — updated each frame so LM starts close to the solution.
    prev_joints = jnp.zeros(robot.joints.num_actuated_joints)
    prev_T_world_root = jaxlie.SE3.identity()

    print("===== RUKA RETARGETING PROCESS =====")
    print("==           Waiting...           ==")
    while True:
        keypoints = subscriber.recv(flags=zmq.NOBLOCK)
        if keypoints is not None:
            T_world_root, joints = solve_retargeting(
                robot=robot,
                target_keypoints=jnp.array(keypoints),
                ruka_link_retarget_indices=ruka_link_idx,
                mano_joint_retarget_indices=mano_joint_idx,
                mano_mask=mano_mask,
                weights=default_weights,
                prev_joints=prev_joints,
                prev_T_world_root=prev_T_world_root,
            )

            prev_joints = joints
            prev_T_world_root = T_world_root

            joints_np = onp.array(joints)
            root_wxyz_xyz = onp.array(T_world_root.wxyz_xyz)
            base_frame.wxyz = root_wxyz_xyz[:4]
            base_frame.position = root_wxyz_xyz[4:]
            urdf_vis.update_cfg(joints_np)

            server.scene.add_point_cloud(
                "/keypoints",
                onp.array(keypoints).reshape(-1, 3),
                colors=onp.full((21, 3), (0, 100, 255), dtype=onp.uint8),
                point_size=0.004,
            )

            publisher.pub(data_array=joints_np, topic_name="ruka_r_joints")

if __name__ == "__main__":
    main()
