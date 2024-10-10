import argparse
import json
import os
import digital_cousins
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler
from digital_cousins.pipeline.generation import SimulatedSceneGenerator
import omnigibson.lazy as lazy


def on_escape_pressed():
    og.clear()
    og.shutdown()

def main(args):
    # Load relevant input information
    with open(args.scene_info_path, "r") as f:
        scene_info = json.load(f)

    h, w = scene_info["resolution"]
    gm.DEFAULT_VIEWER_HEIGHT = h
    gm.DEFAULT_VIEWER_WIDTH = w

    og.launch()
    scene = SimulatedSceneGenerator.load_cousin_scene(scene_info=scene_info, visual_only=True)
    
    # Allow user to teleoperate the camera
    cam_mover = og.sim.enable_viewer_camera_teleoperation()

    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=on_escape_pressed
    )

    # Print out additional keyboard commands
    print(f"\t ESC: Exit")

    # Loop indefinitely
    while True:
        og.sim.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_info_path", type=str, default=os.path.join(os.path.dirname(digital_cousins.ROOT_DIR), "tests/acdc_output/step_3_output/scene_0/scene_0_info.json"),
                        help="Absolute path to acdc output scene info path.")

    args = parser.parse_args()
    main(args)