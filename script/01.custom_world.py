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
from omni.isaac.core import SimulationContext, World
import numpy as np
import argparse

def main(args):
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    # Add Robot Table 
    my_world.scene.add(
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
    my_world.scene.add(
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
            my_world.scene.add(
                DynamicSphere(
                    prim_path="/new_sphere_{}".format(str(i+1)),
                    name="sphere_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.2]),
                    scale=np.array([0.1, 0.1, 0.1]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    linear_velocity=np.array([0, 0, 0.1]),
                ))
        elif args.object_type == 'cube':
            my_world.scene.add(
                DynamicCuboid(
                    prim_path="/new_cube_{}".format(str(i+1)),
                    name="cube_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.2]),
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
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/{}".format(args.robot_type))

    articluated_robot = my_world.scene.add(Robot(prim_path="/World/{}".format(args.robot_type), name="args.robot_type"))
    articluated_robot.set_world_pose(position=np.array([0.2, .0, 1.0]))

    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()
    articluated_robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=init_joint)
    )

    while simulation_app.is_running():
        my_world.step(render=True)
    simulation_app.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--robot_type',     type=str, default='ur5e')
    parser.add_argument('--object_type',     type=str, default='cube')
    args    = parser.parse_args()
    main(args)