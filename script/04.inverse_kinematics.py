from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot 
from omni.isaac.manipulators import SingleManipulator 
from omni.isaac.manipulators.grippers import SurfaceGripper
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader

from omni.isaac.core.objects import cuboid, sphere, capsule, cylinder, cone, ground_plane
from omni.isaac.core import SimulationContext, World
import sys 
sys.path.append('..')
import numpy as np
import argparse
import carb

import math

def rotation_to_quaternion(x_rotation, y_rotation, z_rotation):
    """
    Converts rotation values in the x,y,z axes to a quaternion.
    :param x_rotation: rotation in x-axis (in degrees)
    :param y_rotation: rotation in y-axis (in degrees)
    :param z_rotation: rotation in z-axis (in degrees)
    :return: a tuple containing the quaternion (w, x, y, z)
    """
    roll = math.radians(x_rotation)
    pitch = math.radians(y_rotation)
    yaw = math.radians(z_rotation)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def main_ik_solver(args):
    env = World(stage_units_in_meters=1.0)
    env.scene.add_default_ground_plane()
    # Add a target 
    env.scene.add(
        cuboid.VisualCuboid(
            prim_path="/World/{}".format(args.target_name),
            name=args.target_name,
            position=np.array([0.3, 0, 0.4]),
            scale=np.array([0.07, 0.07, 0.14]),
            color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
        ))
    # Set the root path
    assets_root_path = get_assets_root_path()
    
    if args.robot_type=="Franka":
        robot_path = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        asset_path = assets_root_path + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path="/{}".format(args.robot_type))
        # Gripper 
        ee_frame = "panda_hand"; gripper_attach=False

    elif args.robot_type=="UR5e":
        robot_path = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
        gripper_path = "/Isaac/Robots/Robotiq/2F-85/2f85_instanceable.usd" # "/Isaac/Robots/Robotiq/2F-85/2f85_instanceable.usd"
        asset_path = assets_root_path + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path="/{}".format(args.robot_type))
        # Gripper 
        ee_frame = "flange"; gripper_attach=True 
        gripper_usd = assets_root_path + gripper_path 
        add_reference_to_stage(usd_path=gripper_usd, prim_path="/{}/{}".format(args.robot_type, ee_frame))
        gripper = SurfaceGripper(end_effector_prim_path="/{}/{}".format(args.robot_type, ee_frame), translate=0.1611, direction="z")

    elif args.robot_type=="UR10":
        robot_path = "/Isaac/Robots/UR10/ur10.usd"
        gripper_path = "/Isaac/Robots/UR10/Props/short_gripper.usd" # "/Isaac/Robots/Robotiq/2F-85/2f85_instanceable.usd"
        asset_path = assets_root_path + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path="/{}".format(args.robot_type))
        # Gripper 
        ee_frame = "ee_link"; gripper_attach=True 
        gripper_usd = assets_root_path + gripper_path 
        add_reference_to_stage(usd_path=gripper_usd, prim_path="/{}/{}".format(args.robot_type,ee_frame))
        gripper = SurfaceGripper(end_effector_prim_path="/{}/{}".format(args.robot_type,ee_frame), translate=0.1611, direction="x")

    # Spawn a robot 
    if gripper_attach: robot=env.scene.add(SingleManipulator(prim_path="/{}".format(args.robot_type), name="{}".format(args.robot_type), end_effector_prim_name=ee_frame, gripper=gripper))
    else: robot = env.scene.add(SingleManipulator(prim_path="/{}".format(args.robot_type), name="{}".format(args.robot_type), end_effector_prim_name=ee_frame))
    
    # Set kinematics config  
    kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config(args.robot_type)
    kinematics = LulaKinematicsSolver(**kinematics_config)
    frame_names = kinematics.get_all_frame_names()
    print("{}'s frame names: {}".format(args.robot_type, frame_names))
    offset = np.array([0., 0., 0.])
    robot.set_world_pose(position=offset/ get_stage_units())
    target_orientation = rotation_to_quaternion(180,0,0)
    print("Target quaternion", target_orientation)
    # Initialize Controller 
    controller = ArticulationKinematicsSolver(robot, kinematics, ee_frame)

    # Render 
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()
    while simulation_app.is_running():
        env.step(render=True)
        # IK 
        actions, succ = controller.compute_inverse_kinematics(
            target_position=np.array([0.6, 0., 0.3]),
            target_orientation=target_orientation
        )
        robot.get_articulation_controller().apply_action(actions)
    simulation_app.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Omniverse Usage')
    parser.add_argument('--robot_type',     type=str, default='Franka')
    parser.add_argument('--target_name',     type=str, default='target')

    args    = parser.parse_args()
    main_ik_solver(args)