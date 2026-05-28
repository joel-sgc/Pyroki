"""xArm Gripper Retargeting

Retarget the xArm gripper to OpenPose/MANO palm, thumb, and index keypoints.
The gripper base follows joint 0, the left/right fingers follow joints 4/8,
and the TCP follows the midpoint between joints 4 and 8.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Tuple, TypedDict

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


PALM_KEYPOINT = 0
THUMB_JOINT_2_KEYPOINT = 2
THUMB_TIP_KEYPOINT = 4
INDEX_TIP_KEYPOINT = 8

XARM_GRIPPER_LINKS = {
    "eef": "xarm_gripper_base_link",
    "thumb_knuckle": "left_outer_knuckle",
    "left_tip": "left_tip",
    "right_tip": "right_tip",
    "tcp": "link_tcp",
}


class RetargetingWeights(TypedDict):
    eef_position: float
    """Position weight for placing the gripper base at keypoint 0."""
    tcp_position: float
    """Position weight for placing TCP at midpoint(keypoints 4, 8)."""
    finger_position: float
    """Position weight for placing left/right gripper tips at keypoints 4/8."""
    knuckle_position: float
    """Position weight for placing the thumb-side knuckle at keypoint 2."""
    a_axis: float
    """Direction weight for aligning the left-right gripper axis with keypoint 4-8."""
    d_axis: float
    """Direction weight for aligning the constructed D axes."""
    aperture: float
    """Distance weight for matching gripper opening to keypoint 4-8 distance."""
    joint_smoothness: float
    """Joint smoothness weight."""
    root_smoothness: float
    """Root translation smoothness weight."""


def load_xarm_gripper_urdf(robot_urdf_path: Path) -> yourdfpy.URDF:
    """Load the xArm gripper URDF while resolving relative mesh paths."""
    base_path = robot_urdf_path.parent

    def filename_handler(fname: str) -> str:
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def get_xarm_gripper_link_indices(robot: pk.Robot) -> dict[str, int]:
    """Return link indices used by the gripper retargeting objective."""
    indices = {}
    for key, link_name in XARM_GRIPPER_LINKS.items():
        if link_name not in robot.links.names:
            raise ValueError(f"xArm gripper link '{link_name}' is not present.")
        indices[key] = robot.links.names.index(link_name)
    return indices


def _normalize_np(vec: onp.ndarray) -> onp.ndarray:
    return vec / (onp.linalg.norm(vec) + 1e-6)


def _construct_axes_np(
    left_pos: onp.ndarray,
    right_pos: onp.ndarray,
    midpoint_pos: onp.ndarray,
    eef_or_palm_pos: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray]:
    line_a = _normalize_np(right_pos - left_pos)
    line_b = _normalize_np(midpoint_pos - eef_or_palm_pos)
    line_c = _normalize_np(onp.cross(line_a, line_b))
    line_d = _normalize_np(onp.cross(line_a, line_c))
    return line_a, line_d


def _axis_segment(center: onp.ndarray, axis: onp.ndarray, length: float) -> onp.ndarray:
    half_axis = axis * (length / 2.0)
    return onp.stack([center - half_axis, center + half_axis], axis=0)


def main():
    """Main function for xArm gripper retargeting."""
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
    robot_urdf_path = asset_dir / "xarm" / "xarm_gripper.urdf"

    try:
        urdf = load_xarm_gripper_urdf(robot_urdf_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Expected xArm gripper assets at `examples/retarget_helpers/hand/xarm`."
        ) from exc

    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=jnp.array([0.0]))
    link_indices = get_xarm_gripper_link_indices(robot)

    if args.probe:
        print(f"URDF: {robot_urdf_path}")
        print(f"links: {robot.links.num_links}")
        print(f"joints: {robot.joints.num_joints}")
        print(f"actuated joints: {robot.joints.num_actuated_joints}")
        print("actuated joint order:")
        for i, joint_name in enumerate(robot.joints.actuated_names):
            print(f"  {i:02d}: {joint_name}")
        print("OpenPose/MANO keypoint targets:")
        print(f"  {PALM_KEYPOINT:02d}: {XARM_GRIPPER_LINKS['eef']}")
        print(f"  {THUMB_JOINT_2_KEYPOINT:02d}: {XARM_GRIPPER_LINKS['thumb_knuckle']}")
        print(f"  {THUMB_TIP_KEYPOINT:02d}: {XARM_GRIPPER_LINKS['left_tip']}")
        print(f"  {INDEX_TIP_KEYPOINT:02d}: {XARM_GRIPPER_LINKS['right_tip']}")
        print("  midpoint(04, 08): link_tcp")
        return

    dexycb_motion_path = asset_dir / "dexycb_motion.pkl"
    with open(dexycb_motion_path, "rb") as f:
        dexycb_motion_data = pickle.load(f, encoding="latin1")

    keypoints = dexycb_motion_data["world_hand_joints"]
    assert not onp.isnan(keypoints).any()
    if args.timesteps is not None:
        keypoints = keypoints[: args.timesteps]
    num_timesteps = keypoints.shape[0]

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
        eef_position=0.01,
        tcp_position=30.0,
        finger_position=30.0,
        knuckle_position=5.0,
        a_axis=20.0,
        d_axis=10.0,
        aperture=10.0,
        joint_smoothness=1.0,
        root_smoothness=2.0,
    )

    if args.solve_only:
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=keypoints,
            link_indices=link_indices,
            weights=default_weights,
        )
        print(f"Ts_world_root: {Ts_world_root.wxyz_xyz.shape}")
        print(f"joints: {joints.shape}")
        print(f"actuated joint order: {robot.joints.actuated_names}")
        print(f"joint range: {float(joints.min()):.4f} to {float(joints.max()):.4f}")
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
            link_indices=link_indices,
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

            target_indices = onp.array(
                [PALM_KEYPOINT, THUMB_TIP_KEYPOINT, INDEX_TIP_KEYPOINT]
            )
            target_points = onp.array(keypoints[tstep, target_indices])
            tcp_target = target_points[1:].mean(axis=0, keepdims=True)
            server.scene.add_point_cloud(
                "/scene_offset/gripper_targets",
                onp.concatenate([target_points, tcp_target], axis=0),
                onp.array(
                    [
                        (0, 0, 255),
                        (255, 80, 80),
                        (80, 255, 80),
                        (255, 220, 0),
                    ]
                ),
                point_size=0.008,
                point_shape="sparkle",
            )

            T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=joints[tstep]))
            T_world_root = jaxlie.SE3(Ts_world_root.wxyz_xyz[tstep])
            robot_link_pos = onp.array((T_world_root @ T_root_link).translation())
            left_tip_pos = robot_link_pos[link_indices["left_tip"]]
            right_tip_pos = robot_link_pos[link_indices["right_tip"]]
            tcp_pos = robot_link_pos[link_indices["tcp"]]
            eef_pos = robot_link_pos[link_indices["eef"]]

            target_a_axis, target_d_axis = _construct_axes_np(
                target_points[1],
                target_points[2],
                tcp_target[0],
                target_points[0],
            )
            robot_a_axis, robot_d_axis = _construct_axes_np(
                left_tip_pos,
                right_tip_pos,
                tcp_pos,
                eef_pos,
            )
            axis_length = 0.08
            server.scene.add_line_segments(
                "/scene_offset/orientation_axes",
                points=onp.stack(
                    [
                        onp.stack([target_points[1], target_points[2]], axis=0),
                        onp.stack([left_tip_pos, right_tip_pos], axis=0),
                        _axis_segment(tcp_target[0], target_d_axis, axis_length),
                        _axis_segment(tcp_pos, robot_d_axis, axis_length),
                    ],
                    axis=0,
                ),
                colors=onp.array(
                    [
                        [[0, 70, 255], [0, 70, 255]],
                        [[0, 220, 255], [0, 220, 255]],
                        [[255, 40, 40], [255, 40, 40]],
                        [[255, 150, 0], [255, 150, 0]],
                    ],
                    dtype=onp.uint8,
                ),
                line_width=3.0,
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
    link_indices: dict[str, int],
    weights: RetargetingWeights,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the gripper retargeting problem."""
    timesteps = target_keypoints.shape[0]

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))

    @jaxls.Cost.factory
    def gripper_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()

        eef_pos = link_pos[link_indices["eef"]]
        thumb_knuckle_pos = link_pos[link_indices["thumb_knuckle"]]
        left_tip_pos = link_pos[link_indices["left_tip"]]
        right_tip_pos = link_pos[link_indices["right_tip"]]
        tcp_pos = link_pos[link_indices["tcp"]]

        palm_target = keypoints[PALM_KEYPOINT]
        thumb_knuckle_target = keypoints[THUMB_JOINT_2_KEYPOINT]
        left_target = keypoints[THUMB_TIP_KEYPOINT]
        right_target = keypoints[INDEX_TIP_KEYPOINT]
        tcp_target = (left_target + right_target) / 2.0

        def normalize(vec: jax.Array) -> jax.Array:
            return vec / (jnp.linalg.norm(vec) + 1e-6)

        def construct_d_axis(
            left_pos: jax.Array,
            right_pos: jax.Array,
            midpoint_pos: jax.Array,
            eef_or_palm_pos: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            line_a = normalize(right_pos - left_pos)
            line_b = normalize(midpoint_pos - eef_or_palm_pos)
            line_c = normalize(jnp.cross(line_a, line_b))
            line_d = normalize(jnp.cross(line_a, line_c))
            return line_a, line_d

        target_a_axis, target_d_axis = construct_d_axis(
            left_target,
            right_target,
            tcp_target,
            palm_target,
        )
        robot_a_axis, robot_d_axis = construct_d_axis(
            left_tip_pos,
            right_tip_pos,
            tcp_pos,
            eef_pos,
        )

        target_aperture = jnp.linalg.norm(right_target - left_target)
        robot_aperture = jnp.linalg.norm(right_tip_pos - left_tip_pos)

        return jnp.concatenate(
            [
                (eef_pos - palm_target) * weights["eef_position"],
                (tcp_pos - tcp_target) * weights["tcp_position"],
                (thumb_knuckle_pos - thumb_knuckle_target)
                * weights["knuckle_position"],
                (left_tip_pos - left_target) * weights["finger_position"],
                (right_tip_pos - right_target) * weights["finger_position"],
                (robot_a_axis - target_a_axis) * weights["a_axis"],
                (robot_d_axis - target_d_axis) * weights["d_axis"],
                jnp.array([(robot_aperture - target_aperture) * weights["aperture"]]),
            ],
            axis=0,
        )

    @jaxls.Cost.factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        return (
            var_values[var_Ts_world_root].translation()
            - var_values[var_Ts_world_root_prev].translation()
        ).flatten() * weights["root_smoothness"]

    costs = [
        gripper_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_keypoints,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([weights["joint_smoothness"]]),
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
        ),
    ]

    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[
                var_joints,
                var_Ts_world_root,
            ],
        )
        .analyze()
        .solve()
    )
    return solution[var_Ts_world_root], solution[var_joints]


if __name__ == "__main__":
    main()
