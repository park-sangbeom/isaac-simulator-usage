from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.objects import cuboid, sphere, capsule, cylinder, cone, ground_plane
from omni.isaac.core import SimulationContext, World
import argparse 
import numpy as np 


def main_object_spawn(args):
    env = World(stage_units_in_meters=1.0)
    # Spawn a default plane
    env.scene.add_default_ground_plane() 

    # Add a robot table: Visual cuboid [Fixed]
    env.scene.add(
        cuboid.FixedCuboid(
                    prim_path="/RobotTable",
                    name="RobotTable",
                    position=np.array([0, 0, 0.5]),
                    scale=np.array([0.76, 0.72, 0.79]),
                    size=1.0,
                    color=np.array([102/255, 102/255, 102/255]),
        ))

    # Add Table 
    env.scene.add(
        cuboid.FixedCuboid(
                    prim_path="/Table",
                    name="Table",
                    position=np.array([0.78, 0, 0.37]),
                    scale=np.array([0.8, 0.8, 0.74]),
                    size=1.0,
                    color=np.array([204/255, 102/255, 51/255]),
        ))

    # Random Spawn
    for i in range(args.object_num):
        rand_x = np.random.uniform(low=0.8, high=1.2)
        rand_y = np.random.uniform(low=-0.4, high=0.4)
 
        if args.object_type == 'sphere':
            env.scene.add(
                cuboid.DynamicCuboid(
                    prim_path="/World/new_sphere_{}".format(str(i+1)),
                    name="sphere_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.2]),
                    scale=np.array([0.07, 0.07, 0.07]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    linear_velocity=np.array([0, 0, 0.1]),
                ))
        elif args.object_type == 'cube':
            env.scene.add(
                cuboid.DynamicCuboid(
                    prim_path="/World/new_cube_{}".format(str(i+1)),
                    name="cube_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.1]),
                    scale=np.array([0.07, 0.07, 0.14]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    linear_velocity=np.array([0, 0, 0.1]),
                ))
        elif args.object_type == 'cone':
            env.scene.add(
                cone.DynamicCuboid(
                    prim_path="/World/new_cone_{}".format(str(i+1)),
                    name="cube_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.1]),
                    scale=np.array([1, 1, 1]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    radius=0.07,
                    height=0.14,
                    linear_velocity=np.array([0, 0, 0.1]),
                ))

        elif args.object_type == 'cylinder':
            env.scene.add(
                cylinder.DynamicCylinder(
                    prim_path="/World/new_cylinder_{}".format(str(i+1)),
                    name="cube_{}".format(str(i)),
                    position=np.array([rand_x, rand_y, 1.1]),
                    scale=np.array([1, 1, 1]),
                    color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                    radius=0.07,
                    height=0.14,
                    linear_velocity=np.array([0, 0, 0.1]),
                ))
    # Render 
    while simulation_app.is_running():
        env.step(render=True)
    simulation_app.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Omniverse Usage')
    parser.add_argument('--object_type',     type=str, default='cube')
    parser.add_argument('--object_num',     type=int, default=3)

    args    = parser.parse_args()
    main_object_spawn(args)