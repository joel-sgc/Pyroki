"""xArm Retargeting

Retarget the full xArm plus gripper to OpenPose/MANO palm, thumb, and index
keypoints. The robot base remains fixed at the URDF origin; robot joints and a
single global MANO trajectory offset pose are optimized.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import TypedDict

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
MANO_FLOOR_Z = 0.01

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
    self_collision: float
    """Weight for penalizing robot self-collision."""
    joint_limit: float
    """Weight for softly penalizing joint-limit exceedance."""
    arm_midrange: float
    """Small posture weight that keeps xArm joints 1-7 near their limit midpoints."""
    mano_translation: float
    """Small offset weight that keeps MANO global translation near zero."""
    mano_rotation: float
    """Small offset weight that keeps MANO global rotation near identity."""
    joint_smoothness: float
    """Joint smoothness weight."""
    smooth_beta: float
    """Blend from uniform arm smoothness to proximal-joint-weighted smoothness."""


def load_xarm_urdf(robot_urdf_path: Path) -> yourdfpy.URDF:
    """Load the xArm URDF while resolving relative mesh paths."""
    base_path = robot_urdf_path.parent

    def filename_handler(fname: str) -> str:
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def get_xarm_link_indices(robot: pk.Robot) -> dict[str, int]:
    """Return link indices used by the retargeting objective."""
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


def _transform_mano_trajectory_np(
    transform: jaxlie.SE3,
    keypoints_raw: onp.ndarray,
    hand_vertices_raw: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray]:
    transform = transform.normalize()
    keypoints_transformed = transform.apply(jnp.array(keypoints_raw))
    hand_vertices_transformed = transform.apply(jnp.array(hand_vertices_raw))
    return onp.array(keypoints_transformed), onp.array(hand_vertices_transformed)


def _lift_transform_to_mano_floor(
    transform: jaxlie.SE3,
    keypoints: jax.Array,
) -> jaxlie.SE3:
    transform = transform.normalize()
    min_keypoint_z = transform.apply(keypoints)[..., 2].min()
    z_lift = jnp.maximum(0.0, MANO_FLOOR_Z - min_keypoint_z)
    return jaxlie.SE3.from_rotation_and_translation(
        transform.rotation(),
        transform.translation() + jnp.array([0.0, 0.0, z_lift]),
    ).normalize()


def main():
    """Main function for full xArm retargeting."""
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
    robot_urdf_path = asset_dir / "xarm" / "xarm7_standalone.urdf"

    try:
        urdf = load_xarm_urdf(robot_urdf_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Expected xArm assets at `examples/retarget_helpers/hand/xarm`."
        ) from exc

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    link_indices = get_xarm_link_indices(robot)

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

    keypoints_raw = dexycb_motion_data["world_hand_joints"]
    assert not onp.isnan(keypoints_raw).any()
    if args.timesteps is not None:
        keypoints_raw = keypoints_raw[: args.timesteps]
    hand_vertices_raw = dexycb_motion_data["world_hand_vertices"]
    if args.timesteps is not None:
        hand_vertices_raw = hand_vertices_raw[: args.timesteps]
    mano_origin = keypoints_raw[0, PALM_KEYPOINT].copy()
    default_mano_translation = onp.array([-mano_origin[0], -mano_origin[1], 0.0])
    default_T_world_mano = jaxlie.SE3.from_translation(
        jnp.array(default_mano_translation)
    )

    keypoints, hand_vertices = _transform_mano_trajectory_np(
        default_T_world_mano,
        keypoints_raw,
        hand_vertices_raw,
    )
    num_timesteps = keypoints_raw.shape[0]

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
        finger_position=50.0,
        knuckle_position=0.01,
        a_axis=40.0,
        d_axis=30.0,
        aperture=10.0,
        self_collision=80.0,
        joint_limit=80.0,
        arm_midrange=10.0,
        mano_translation=0.01,
        mano_rotation=0.01,
        joint_smoothness=12.0,
        smooth_beta=0.0,
    )

    if args.solve_only:
        joints, T_world_mano = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=keypoints_raw,
            link_indices=link_indices,
            default_T_world_mano=default_T_world_mano,
            weights=default_weights,
        )
        shifted_initial_palm = onp.array(
            T_world_mano.apply(jnp.array(keypoints_raw[0, PALM_KEYPOINT]))
        )
        transformed_keypoints = T_world_mano.apply(jnp.array(keypoints_raw))
        transformed_vertices = T_world_mano.apply(jnp.array(hand_vertices_raw))
        raw_vertex_distances = jnp.linalg.norm(
            jnp.array(hand_vertices_raw[0, 1:]) - jnp.array(hand_vertices_raw[0, :1]),
            axis=-1,
        )
        transformed_vertex_distances = jnp.linalg.norm(
            transformed_vertices[0, 1:] - transformed_vertices[0, :1],
            axis=-1,
        )
        max_distance_error = jnp.max(
            jnp.abs(transformed_vertex_distances - raw_vertex_distances)
        )
        print(f"joints: {joints.shape}")
        print(f"mano transform wxyz_xyz: {onp.array(T_world_mano.wxyz_xyz)}")
        print(f"mano quaternion norm: {float(jnp.linalg.norm(T_world_mano.wxyz_xyz[:4]))}")
        print(f"shifted initial palm: {shifted_initial_palm}")
        print(f"min MANO keypoint z: {float(transformed_keypoints[..., 2].min())}")
        print(f"max MANO vertex distance error: {float(max_distance_error)}")
        print(f"actuated joint order: {robot.joints.actuated_names}")
        print(f"joint min: {onp.array(joints).min(axis=0)}")
        print(f"joint max: {onp.array(joints).max(axis=0)}")
        return

    server = viser.ViserServer()

    hand_mesh = server.scene.add_mesh_simple(
        "/hand_mesh",
        vertices=hand_vertices[0, :, :],
        faces=dexycb_motion_data["hand_mesh_faces"],
        opacity=0.35,
    )
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    object_handle = server.scene.add_mesh_trimesh("/object", mesh)
    server.scene.add_grid("/grid", 2.0, 2.0)

    weights = pk.viewer.WeightTuner(
        server,
        default_weights,  # type: ignore
    )

    joints = None
    T_world_mano = None

    def generate_trajectory():
        nonlocal hand_vertices, joints, keypoints, T_world_mano
        gen_button.disabled = True
        joints, T_world_mano = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=keypoints_raw,
            link_indices=link_indices,
            default_T_world_mano=default_T_world_mano,
            weights=weights.get_weights(),  # type: ignore
        )
        keypoints, hand_vertices = _transform_mano_trajectory_np(
            T_world_mano,
            keypoints_raw,
            hand_vertices_raw,
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    generate_trajectory()
    assert joints is not None and T_world_mano is not None

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array([1.0, 0.0, 0.0, 0.0])
            base_frame.position = onp.zeros(3)
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            target_indices = onp.array(
                [PALM_KEYPOINT, THUMB_TIP_KEYPOINT, INDEX_TIP_KEYPOINT]
            )
            target_points = onp.array(keypoints[tstep, target_indices])
            tcp_target = target_points[1:].mean(axis=0, keepdims=True)
            server.scene.add_point_cloud(
                "/gripper_targets",
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
            robot_link_pos = onp.array(T_root_link.translation())
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
                "/orientation_axes",
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
                "/contact_points",
                onp.array(contact_points_per_frame[tstep]).reshape(-1, 3),
                onp.array((255, 0, 0))[None]
                .repeat(len(contact_points_per_frame[tstep]), axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="circle",
            )
            hand_mesh.vertices = hand_vertices[tstep, :, :]
            object_handle.position = object_pose_list[tstep][:3, 3]
            object_handle.wxyz = R.from_matrix(object_pose_list[tstep][:3, :3]).as_quat(
                scalar_first=True
            )

        time.sleep(0.05)


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_keypoints: jnp.ndarray,
    link_indices: dict[str, int],
    default_T_world_mano: jaxlie.SE3,
    weights: RetargetingWeights,
) -> tuple[jnp.ndarray, jaxlie.SE3]:
    """Solve the full-arm retargeting problem with a fixed robot base."""
    timesteps = target_keypoints.shape[0]

    class ManoTransformVar(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: default_T_world_mano,
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_T_world_mano = ManoTransformVar(jnp.array(0))

    @jaxls.Cost.factory(kind="constraint_geq_zero", name="mano_floor_constraint")
    def mano_floor_constraint(
        var_values: jaxls.VarValues,
        var_T_world_mano: ManoTransformVar,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        keypoints = var_values[var_T_world_mano].normalize().apply(keypoints)
        return keypoints[:, 2] - MANO_FLOOR_Z

    @jaxls.Cost.factory
    def mano_offset_regularization_cost(
        var_values: jaxls.VarValues,
        var_T_world_mano: ManoTransformVar,
    ) -> jax.Array:
        T_world_mano = var_values[var_T_world_mano].normalize()
        return jnp.concatenate(
            [
                T_world_mano.translation() * weights["mano_translation"],
                T_world_mano.rotation().log() * weights["mano_rotation"],
            ],
            axis=0,
        )

    @jaxls.Cost.factory
    def arm_midrange_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        arm_lower = robot.joints.lower_limits[:7]
        arm_upper = robot.joints.upper_limits[:7]
        arm_mid = (arm_lower + arm_upper) / 2.0
        arm_half_range = (arm_upper - arm_lower) / 2.0
        return (
            ((robot_cfg[:7] - arm_mid) / (arm_half_range + 1e-6))
            * weights["arm_midrange"]
        )

    def smoothness_weight_vector() -> jax.Array:
        beta = jnp.clip(weights["smooth_beta"], 0.0, 1.0)
        uniform_arm = jnp.ones(7)
        proximal_arm = jnp.array([4.0, 3.0, 1.1, 0.9, 0.7, 0.2, 0.1])
        proximal_arm = proximal_arm / jnp.mean(proximal_arm)
        arm_weights = (1.0 - beta) * uniform_arm + beta * proximal_arm
        drive_weight = jnp.array([1.0])
        return jnp.concatenate([arm_weights, drive_weight]) * weights[
            "joint_smoothness"
        ]

    @jaxls.Cost.factory
    def gripper_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_T_world_mano: ManoTransformVar,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        keypoints = var_values[var_T_world_mano].normalize().apply(keypoints)
        T_world_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
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

    costs = [
        gripper_alignment_cost(
            var_joints,
            var_T_world_mano,
            target_keypoints,
        ),
        mano_floor_constraint(
            var_T_world_mano,
            target_keypoints,
        ),
        mano_offset_regularization_cost(var_T_world_mano),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.02,
            weight=jnp.array([weights["self_collision"]]),
        ),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            weight=jnp.array([weights["joint_limit"]]),
        ),
        arm_midrange_cost(var_joints),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            smoothness_weight_vector()[None],
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
                var_T_world_mano,
            ],
        )
        .analyze()
        .solve(
            termination=jaxls.TerminationConfig(
                max_iterations=200,
                early_termination=True,
            ),
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(
                tolerance_absolute=1e-7,
                tolerance_relative=1e-6,
                penalty_max=1e9,
                inner_solve_tolerance=1e-4,
            ),
        )
    )
    T_world_mano = _lift_transform_to_mano_floor(
        solution[var_T_world_mano],
        target_keypoints,
    )
    return solution[var_joints], T_world_mano


if __name__ == "__main__":
    main()
