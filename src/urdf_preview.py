"""Simple URDF previewer with viser.

Loads the mirrored RUKA URDF and gives you one slider per actuated joint
so you can test the movement of each joint interactively.

Run from anywhere:
    python view_urdf.py
Then open the printed viser URL (default http://localhost:8080) in a browser.
"""

from pathlib import Path

import numpy as onp
import viser
import yourdfpy
from viser.extras import ViserUrdf

# --- Path setup (script lives in ~/pyroki/src, assets in ~/pyroki/examples) ---
REPO_ROOT = Path(__file__).resolve().parents[1]  # ~/pyroki
ROBOT_URDF_PATH = (
    REPO_ROOT / "examples" / "retarget_helpers" / "hand" / "ruka" / "robot_mirrored.urdf"
)


def load_urdf(robot_urdf_path: Path) -> yourdfpy.URDF:
    """Load the URDF while resolving its package://assets mesh paths."""
    base_path = robot_urdf_path.parent

    def filename_handler(fname: str) -> str:
        if fname.startswith("package://assets/"):
            fname = fname.removeprefix("package://assets/")
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def main() -> None:
    assert ROBOT_URDF_PATH.exists(), f"URDF not found: {ROBOT_URDF_PATH}"
    urdf = load_urdf(ROBOT_URDF_PATH)

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=0.5, height=0.5, cell_size=0.05)
    server.scene.add_frame("/base", show_axes=True, axes_length=0.05, axes_radius=0.002)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # One slider per actuated joint, bounds taken from the URDF limits.
    sliders: list[viser.GuiInputHandle[float]] = []
    initial_cfg: list[float] = []

    def update_cfg(_=None) -> None:
        urdf_vis.update_cfg(onp.array([s.value for s in sliders]))

    with server.gui.add_folder("Joints"):
        for joint_name, (lower, upper) in urdf_vis.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -onp.pi
            upper = upper if upper is not None else onp.pi
            init = float(onp.clip(0.0, lower, upper))
            slider = server.gui.add_slider(
                label=joint_name,
                min=float(lower),
                max=float(upper),
                step=1e-3,
                initial_value=init,
            )
            slider.on_update(update_cfg)
            sliders.append(slider)
            initial_cfg.append(init)

    reset_button = server.gui.add_button("Reset joints")

    @reset_button.on_click
    def _(_) -> None:
        for s, init in zip(sliders, initial_cfg):
            s.value = init  # each assignment triggers update_cfg via on_update

    update_cfg()
    print(f"Loaded {ROBOT_URDF_PATH.name} with {len(sliders)} actuated joints.")
    print("Viewer running — press Ctrl+C to exit.")
    server.sleep_forever()


if __name__ == "__main__":
    main()