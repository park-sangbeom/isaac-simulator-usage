
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
import sys 
sys.path.append('..')
from env.omni_env import OmniverseEnvironment
from env.omni_base import OmniBase
from omni.isaac.core import SimulationContext
import yaml 
from pathlib import Path 
import os 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import copy 
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.manipulators import SingleManipulator 
from omni.isaac.franka.tasks import Stacking
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.objects import DynamicCuboid

class FrankaWorld(OmniBase):
    def __init__(self, name="Franka World", map_name='grid_default'):
        self.env = OmniverseEnvironment(map_name=map_name)

    def set_env(self):
        self.env.add_env()

    def set_object(self,object_info):    
        self.env.add_object(object_info=object_info)
    
    def set_robot(self,robot_info):
        self.env.add_robot(robot_info=robot_info)
        # self.env.add_controller(robot_name=robot_info["name"])

    def set_camera(self, camera_info):
        self.env.add_camera(prim_path=camera_info["prim_path"],
                            position=camera_info["position"],
                            rotation=camera_info["rotation"],
                            resolution=camera_info["resolution"])

    def get_rgb(self, rel_dir="omniverse_usage/script/data/npy/", image_name="test.npy"):
        for camera_idx, camera in enumerate(self.env.camera_lst):
            rgb = camera.get_rgba()[:,:,:3]
            path = Path.joinpath(Path.cwd(), rel_dir, str(camera_idx)+image_name)
            np.save(path, rgb)
        return rgb 
    
if __name__=="__main__": 
    # World 
    task = FrankaWorld(map_name='grid_default')
    # YAML
    task_info_dir = Path.joinpath(Path.cwd(), 'omniverse_usage/script/cfg/','pnp_task.yaml')
    with open(task_info_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Set Plane
    task.set_env()
    # Robot, Object, Camera Setup
    for agent_idx in range(config["agent_num"]):
        # Robot
        robot_info = {"name": config["robot_info"]["name"]+"{}".format(agent_idx+1),
                    "prim_path": config["robot_info"]["prim_path"]+"{}".format(agent_idx+1),
                    "end_effector_prim_path": "/World/Franka{}/panda_rightfinger".format(agent_idx+1),
                    "joint_prim_names": config["robot_info"]["joint_prim_names"],
                    "robot_path": config["robot_info"]["robot_path"],
                    "translation_offset": np.array([0,0+agent_idx*5,0]),
                    "end_effector_prim_name": config["robot_info"]["end_effector_prim_name"]}
        task.set_robot(robot_info=robot_info)
        # Camera 
        camera_info = {"prim_path": config["camera_info"]["prim_path"]+"{}".format(agent_idx+1),
                    "position": np.array([1.5,0+agent_idx*5,1.3]),
                    "rotation": np.array([180,135,0]),
                    "resolution": (256, 256)}
        task.set_camera(camera_info=camera_info)
        # Object 
        for object_idx in range(config["object_num"]):
            object_info = {"name": config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1), 
                        "prim_path": config["object_info"]["prim_path"]+"_{}_{}".format(agent_idx+1, object_idx+1),
                        "position": np.array([0.5+object_idx*0.1,0+agent_idx*5,0.3]),
                        "scale": np.array([0.07, 0.07, 0.14]),
                        "type": config["object_info"]["type"]}
            task.set_object(object_info=object_info)

    task.env.world.reset()
    # task.env.camera.initialize()
    # task.env.camera.add_motion_vectors_to_frame()

    # Simulation Init 
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()

    while simulation_app.is_running():
        task.env.world.step(render=True) 
        rgb = task.get_rgb()
        obs = task.env.get_observations()
        # for control_idx, (controller,articulation) in enumerate(zip(task.env.controller_lst, task.env.articulation_lst)): 
        #     franka = task.env.world.scene.get_object("Franka{}".format(control_idx+1))
        #     print("Franka{}".format(control_idx+1))
        #     for obj_idx in range(config["object_num"]):
        #         print(obs["object"+"_{}_{}".format(control_idx+1, obj_idx+1)]["position"]+np.array([0,0.2,0]))
        #         actions = controller.forward(
        #             picking_position=obs["object"+"_{}_{}".format(control_idx+1, obj_idx+1)]["position"]-np.array([obj_idx*0.1, agent_idx*5, 0]),
        #             placing_position=obs["object"+"_{}_{}".format(control_idx+1, obj_idx+1)]["position"]-np.array([obj_idx*0.1, agent_idx*5, 0])+np.array([0,0.2,0]),
        #             current_joint_positions=franka.get_joint_positions(),
        #             end_effector_offset=np.array([0, 0.005, 0.005]))
        #         print(actions)
        #         articulation.apply_action(actions)
    simulation_app.close()


    # robot_path = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
    # asset_path = get_assets_root_path() + robot_path
    # add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
    # gripper = ParallelGripper(
    #     end_effector_prim_path="/World/Franka/panda_rightfinger",
    #     joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    #     joint_opened_positions=np.array([0.05, 0.05]),
    #     joint_closed_positions=np.array([0.0, 0.0]),
    #     action_deltas=np.array([0.05, 0.05]),
    # )
    # cube = task.env.world.scene.add(
    #     DynamicCuboid(
    #         prim_path="/World/random_cube",
    #         name="object",
    #         position=np.array([0.5, 0, 0.1]),
    #         scale=np.array([0.07, 0.07, 0.14]),
    #         color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
    #     )
    # )
    # robot = task.env.world.scene.add(
    #     SingleManipulator(prim_path="/World/Franka", name="{}".format("Franka"), end_effector_prim_name='panda_rightfinger', gripper=gripper)
    # )
    # robot = task.env.world.scene.get_object("Franka")
    # robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    # controller =  PickPlaceController(name="pick_place_controller", gripper=robot.gripper, robot_articulation=robot)
    # articulation_controller = robot.get_articulation_controller()

    # while simulation_app.is_running():
    #     task.env.world.step(render=True) 
    #     # rgb = task.get_rgb()
    #     # obs = task.env.get_observations()
    #     actions = controller.forward(
    #         picking_position=task.env.world.scene.get_object(name='object').get_local_pose()[0],
    #         placing_position=np.array([0.5,0.3,0.07]),
    #         current_joint_positions=robot.get_joint_positions(),
    #         end_effector_offset=np.array([0, 0.005, 0.005]))
    #     articulation_controller.apply_action(actions)
    #     if controller.is_done():
    #         print("Controller is done")
    # simulation_app.close()
