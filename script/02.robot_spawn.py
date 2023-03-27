from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot 
from omni.isaac.core import SimulationContext, World
import sys 
sys.path.append('..')
import numpy as np
import argparse
import carb

def main_robot_spawn(args):
    env = World(stage_units_in_meters=1.0)
    env.scene.add_default_ground_plane()
            
    # Add a robot
    if args.robot_type=="franka":
        robot_path = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        init_joint = np.array([0, 0, 0, -1.5, -0.5,1.5,1.5,0,0])
    elif args.robot_type=="ur5e":
        robot_path = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
        init_joint = np.array([0, -1.5, 1.5, 1, 0, 0])
    elif args.robot_type=="ur10":
        robot_path = "/Isaac/Robots/UR10/ur10.usd"
        init_joint = np.array([0, -1.5, 1.5, 1, 0, 0])

    # Set the root path
    assets_root_path = get_assets_root_path()
    asset_path = assets_root_path + robot_path
    add_reference_to_stage(usd_path=asset_path, prim_path="/{}".format(args.robot_type))
    offset = np.array([0., 0., 0.])
    robot = env.scene.add(Robot(prim_path="/{}".format(args.robot_type), name="{}".format(args.robot_type)))
    robot.set_world_pose(position=offset/ get_stage_units())

    # Initialize
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()
    
    # Init action 
    robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=init_joint)
    )

    # Render 
    while simulation_app.is_running():
        env.step(render=True)
    simulation_app.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Omniverse Usage')
    parser.add_argument('--robot_type',     type=str, default='ur5e')
    args    = parser.parse_args()
    main_robot_spawn(args)