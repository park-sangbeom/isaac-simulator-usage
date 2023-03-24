from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import VisualCuboid, DynamicCuboid
from omni.isaac.core.objects.sphere import VisualSphere, DynamicSphere, FixedSphere
from omni.isaac.core.objects.cylinder import VisualCylinder, DynamicCylinder, FixedCylinder
from omni.isaac.core.objects.cone import VisualCone, DynamicCone, FixedCone
from omni.isaac.core.objects.capsule import VisualCapsule, DynamicCapsule, FixedCapsule
from omni.isaac.core.robots import Robot 
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core import SimulationContext, World
import sys 
sys.path.append('..')
import numpy as np
import argparse
import carb
from model.mjcf_parser import MJCFParserClass,r2w,rpy2r,quaternion_rotation_matrix

def main(args):
    mjcf_parser = MJCFParserClass(rel_xml_path='./omniverse_usage/script/asset/ur5e_rg2/ur5e_rg2.xml')
    
    env = World(stage_units_in_meters=1.0)
    env.scene.add_default_ground_plane()
    # Add Robot Table 
    env.scene.add(
        DynamicCuboid(
                    prim_path="/RobotTable",
                    name="RobotTable",
                    position=np.array([0, 0, 0.5]),
                    scale=np.array([1, 1, 1]),
                    size=1.0,
                    color=np.array([102/255, 102/255, 102/255]),
                    linear_velocity=np.array([0, 0, 0]),
        ))

    # Add Table 
    env.scene.add(
        DynamicCuboid(
                    prim_path="/Table",
                    name="Table",
                    position=np.array([0.801, 0, 0.45]),
                    scale=np.array([0.6, 0.8, 0.9]),
                    size=1.0,
                    color=np.array([204/255, 102/255, 51/255]),
                    linear_velocity=np.array([0, 0, 0]),
        ))

    # Add objects 
    for i in range(3):
        rand_x = np.random.uniform(low=0.75, high=0.9)
        rand_y = np.random.uniform(low=-0.4, high=0.4)
        if args.object_type == 'sphere':
            env.scene.add(
                DynamicSphere(
                    prim_path="/World/new_sphere_{}".format(str(i+1)),
                    name="sphere_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.2]),
                    scale=np.array([0.1, 0.1, 0.1]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    linear_velocity=np.array([0, 0, 0.1]),
                ))
        elif args.object_type == 'cube':
            env.scene.add(
                DynamicCuboid(
                    prim_path="/World/new_cube_{}".format(str(i+1)),
                    name="cube_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.1]),
                    scale=np.array([0.1, 0.1, 0.1]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    linear_velocity=np.array([0, 0, 0.1]),
                ))

            
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

    assets_root_path = get_assets_root_path()
    asset_path = assets_root_path + robot_path
    add_reference_to_stage(usd_path=asset_path, prim_path="/{}".format(args.robot_type))
    offset = np.array([0.2, .0, 1.0])
    robot = env.scene.add(Robot(prim_path="/{}".format(args.robot_type), name="args.robot_type"))
    robot.set_world_pose(position=offset/ get_stage_units())

    # Simple Inverse Kinematics
    controller = KinematicsSolver(robot)
    articulation_controller = robot.get_articulation_controller()
    target_name = 'cube_2'

    # Run
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()
    
    # Init action 
    # robot.get_articulation_controller().apply_action(
    # ArticulationAction(joint_positions=init_joint)
    # )

    # Get observations 
    cnt=0 
    while simulation_app.is_running():
        cnt+=1
        if cnt==500: 
            target_obj = env.scene.get_object(target_name)
            target_position,target_rotation = target_obj.get_local_pose()
            print("target_position", target_position)
            print("target_rotation",quaternion_rotation_matrix(target_rotation))
            q,_ = mjcf_parser.onestep_ik(body_name='tcp_link', p_trgt=target_position, R_trgt=quaternion_rotation_matrix(target_rotation), IK_P=True, IK_R=True,
                                       joint_idxs=mjcf_parser.rev_joint_idxs)
            print("q", q)
            robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=q[:6])
            )

        env.step(render=True)
    simulation_app.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--robot_type',     type=str, default='ur5e')
    parser.add_argument('--object_type',     type=str, default='cube')
    args    = parser.parse_args()
    main(args)