from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core import World
import omni.isaac.core.utils.numpy.rotations as rot_utils
import numpy as np
import matplotlib.pyplot as plt


my_world = World(stage_units_in_meters=1.0)

cube_2 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/new_cube_2",
        name="cube_1",
        position=np.array([.0, 3, 1.0]),
        scale=np.array([0.6, 0.5, 0.2]),
        size=1.0,
        color=np.array([255, 0, 0]),
    )
)

cube_3 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/new_cube_3",
        name="cube_2",
        position=np.array([2, 1, 3.0]),
        scale=np.array([1, 1, 1]),
        size=1.0,
        color=np.array([0, 0, 255]),
        linear_velocity=np.array([0, 0, 0.4]),
    )
)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 25.0]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)

my_world.scene.add_default_ground_plane()
my_world.reset()
camera.initialize()

i = 0
camera.add_motion_vectors_to_frame()

while simulation_app.is_running():
    my_world.step(render=True)
    print(i)
    # print(camera.get_current_frame())
    if i == 10:
        points_2d = camera.get_image_coords_from_world_points(
            np.array([cube_3.get_world_pose()[0], cube_2.get_world_pose()[0]])
        )
        points_3d = camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
        print(points_2d)
        print(points_3d)
        imgplot = plt.imshow(camera.get_rgba()[:, :, :3])
        plt.show()
        plt.savefig('test.png')
        print(camera.get_current_frame()["motion_vectors"])
        np.save('test.npy', camera.get_rgba()[:, :, :3])
        break 
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
    i += 1


simulation_app.close()