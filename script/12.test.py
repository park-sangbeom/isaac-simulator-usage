
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
import sys 
sys.path.append('..')
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
from omni.isaac.franka.controllers import StackingController
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.franka.controllers import RMPFlowController

from omni.isaac.core.objects import cuboid, sphere, capsule, cylinder, cone, ground_plane
from omni.isaac.core import World
from typing import Optional
import carb 

class FrankSim(OmniBase):
    def __init__(self, task_name="pick-and-place", plane_name='grid_default'):
        self.env = World()
        self.task_name  = task_name 
        self.plane_name = plane_name
        self.object_lst       = []
        self.robot_lst        = []
        self.controller_lst   = []
        self.articulation_lst = []
        self.camera_lst       = []
        self.simulation_context = SimulationContext()

    def init_simulation(self):
        self.simulation_context.initialize_physics()
        self.simulation_context.play()

    def reset_controller(self):
        self.controller_lst = []

    def add_plane(self):
        asset_path=""
        if self.plane_name=="grid_default":
            asset_path = get_assets_root_path()+"/Isaac/Environments/Grid/default_environment.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")

        elif self.plane_name=="simple_room":
            asset_path = get_assets_root_path()+"/Isaac/Environments/Simple_Room/simple_room.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")

        elif self.plane_name=="office":
            asset_path = get_assets_root_path()+"/Isaac/Environments/Office/office.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        
        else: 
            carb.log_error("Could not find the '{}' environment".format(self.plane_name))

    def add_robot(self, robot_info=None, object_order_lst=None):
        robot_path = robot_info["robot_path"]
        asset_path = get_assets_root_path() + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path=robot_info["prim_path"])
        gripper=ParallelGripper(
            end_effector_prim_path=robot_info["end_effector_prim_path"],
            joint_prim_names=robot_info["joint_prim_names"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.05, 0.05]))
        # Robot 
        robot=self.env.scene.add((
            SingleManipulator(
            prim_path=robot_info["prim_path"], 
            name=robot_info["name"], 
            end_effector_prim_name=robot_info["end_effector_prim_name"],
            gripper=gripper,
            position=robot_info["translation_offset"])))
        robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
        self.robot_lst.append(robot_info["name"])
    
    def add_controller(self):
        for robot_name in self.robot_lst: 
            robot = self.env.scene.get_object(robot_name)
            # Controller Setup
            if self.task_name=="pick-and-place":
                self.controller =  PickPlaceController(name="pick_place_controller", gripper=robot.gripper, robot_articulation=robot)
            elif self.task_name=="following":
                self.controller =  RMPFlowController(name="target_follower_controller", robot_articulation=robot)
            elif self.task_name=="stacking": 
                self.controller = StackingController(
                    name="stacking_controller",
                    gripper=robot.gripper,
                    robot_articulation=robot,
                    picking_order_cube_names=object_order_lst,
                    robot_observation_name=robot.name)
            self.controller_lst.append(self.controller)
            # Articulation controller
            self.articulation_controller = robot.get_articulation_controller()
            self.articulation_lst.append(self.articulation_controller)
        self.max_count = len(self.controller_lst)

    def add_object(self, object_info=None):
        if object_info["type"] == "fixed cuboid":
            self.env.scene.add(cuboid.FixedCuboid( 
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)])
                                ))
        
        elif object_info["type"]=="dynamic cuboid":
            self.env.scene.add(cuboid.DynamicCuboid(
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                                ))
        elif object_info["type"]=="visual cuboid": 
            self.env.scene.add(cuboid.VisualCuboid(
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                                ))
        self.object_lst.append(object_info["name"])

    def add_camera(self, camera_info):
        self.camera = Camera(
            prim_path=camera_info["prim_path"],
            position=camera_info["position"],
            frequency=20,
            resolution=camera_info["resolution"],
            orientation=rot_utils.euler_angles_to_quats(camera_info["rotation"], degrees=True))
        self.camera_lst.append(self.camera)

    def get_rgb(self, rel_dir="omniverse_usage/script/data/npy/", image_name="test.npy"):
        for camera_idx, camera in enumerate(self.camera_lst):
            rgb = camera.get_rgba()[:,:,:3]
            path = Path.joinpath(Path.cwd(), rel_dir, str(camera_idx)+image_name)
            np.save(path, rgb)
        return rgb 
    
    def get_observations(self):
        self.observations = dict()
        # Object observation
        for obj_name in self.object_lst:
            obj = self.env.scene.get_object(name=obj_name)
            self.observations.update({obj_name: {"position":obj.get_local_pose()[0], "orientation":obj.get_local_pose()[1], "target_position":np.array([0.8, 0.2, 0.07])}})
        # Robot observation
        for robot_name in self.robot_lst:
            robo = self.env.scene.get_object(name=robot_name)
            self.observations.update({robot_name: {"joint_positions":robo.get_joints_state().positions,"end_effector_position":robo.end_effector.get_local_pose()[0]}})
        return self.observations
    

    def follow_target(self):
        """
            Following a target object Task
        """
        self.init_simulation()

        # Set target position 
        target_position_lst = []
        for _agent_idx in range(len(self.robot_lst)): 
            target_position_lst.append(np.array([np.random.uniform(0.5,0.7), np.random.uniform(-0.3,0.3)+_agent_idx*2.0, 0.1]))

        while simulation_app.is_running():
            self.env.step(render=True)
            obs = self.get_observations()
            for agent_idx, (robot_name, controller, articulation_controller, target_position) in enumerate(zip(self.robot_lst, self.controller_lst, self.articulation_lst, target_position_lst)):
                target_name="object_{}_1".format(agent_idx+1)
                actions = controller.forward(
                    target_end_effector_position=obs[target_name]["position"],
                    target_end_effector_orientation=obs[target_name]["orientation"])
                articulation_controller.apply_action(actions)
        simulation_app.close()

    def stack(self, total_object_lst=None): 
        """
        Stakcing objects Task
        """
        self.init_simulation()

        # Set target position 
        place_position_lst = []
        for _agent_idx in range(len(self.robot_lst)): 
            place_position_lst.append(np.array([np.random.uniform(0.5,0.7), np.random.uniform(-0.3,0.3)+_agent_idx*2.0, 0.1]))
        while simulation_app.is_running():
            self.env.step(render=True)
            obs = self.get_observations()
            for agent_idx, (controller, articulation_controller) in enumerate(zip(self.controller_lst, self.articulation_lst)):
                actions = controller.forward(obs)
                articulation_controller.apply_action(actions)
                # if self._controller.is_done():
                #     print("Controller is done")
                #     self.env.pause()
                #     self.env.reset()
                #     self.world_cleanup()
                #     break 
        simulation_app.close()


    def pick_and_place(self, epochs=1): 
        """
            Pick and Place Task
        """
        self.init_simulation()

        for _ in range(epochs):
            reset_count=0 
            # Set place position
            place_position_lst = []
            for _agent_idx in range(len(self.robot_lst)): 
                place_position_lst.append(np.array([np.random.uniform(0.5,0.7), np.random.uniform(-0.3,0.3)+_agent_idx*2.0, 0.1]))
            self.add_controller()
            while simulation_app.is_running():
                self.env.step(render=True)
                obs = self.get_observations()
                self.get_rgb()
                for agent_idx, (robot_name, controller, articulation_controller, place_position) in enumerate(zip(self.robot_lst, self.controller_lst, self.articulation_lst, place_position_lst)):
                    target_object = "object_{}_1".format(format(agent_idx+1))
                    my_franka = self.env.scene.get_object(robot_name)
                    actions = controller.forward(
                            picking_position=obs[target_object]["position"],
                            placing_position=place_position,
                            current_joint_positions=my_franka.get_joint_positions(),
                            end_effector_offset=np.array([0, 0.005, 0.005]))
                    articulation_controller.apply_action(actions)
                    if controller.is_done():
                        print("Controller is done")
                        reset_count+=1 
                if reset_count == self.max_count:
                    self.env.pause()
                    self.env.reset()
                    self.reset_controller()
                    reset_count=0
                    break 
        simulation_app.close()

    def main(self): 
        if self.task_name=="pick-and-place":
            self.pick_and_place()
        elif self.task_name=="following":
            self.follow_target()
        elif self.task_name=="stacking":
            self.stack()


if __name__=="__main__": 
    # YAML
    task_info_dir = Path.joinpath(Path.cwd(), 'omniverse_usage/script/cfg/','pnp_task.yaml')
    with open(task_info_dir, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    # Task environment  
    task_name = 'pick-and-place'
    task = FrankSim(task_name=task_name,plane_name='grid_default')
    # Set Plane
    task.add_plane()
    # Robot, Object, Camera Setup
    # Total stacking object lst = []
    total_object_lst = []
    for agent_idx in range(config["agent_num"]):
        # Camera 
        camera_info = {"prim_path": config["camera_info"]["prim_path"]+"{}".format(agent_idx+1),
                       "position": np.array([1.8,0+agent_idx*config["place_offset"],1.4]),
                       "rotation": np.array(config["camera_info"]["rotation"]),
                       "resolution": config["camera_info"]["resolution"]}
        task.add_camera(camera_info=camera_info)
        # Object 
        object_order_lst = []
        for object_idx in range(config["object_num"]):
            object_info = {"name": config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1), 
                           "prim_path": config["object_info"]["prim_path"]+"_{}_{}".format(agent_idx+1, object_idx+1),
                           "position": np.array([0.5+object_idx*0.1,0+agent_idx*config["place_offset"],0.3]),
                           "scale": np.array(config["object_info"]["scale"]),
                           "type": config["object_info"]["type"]}
            object_order_lst.append(config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1))
            task.add_object(object_info=object_info)
        # Robot
        robot_info = {"name": config["robot_info"]["name"]+"{}".format(agent_idx+1),
                    "prim_path": config["robot_info"]["prim_path"]+"{}".format(agent_idx+1),
                    "end_effector_prim_path": "/World/Franka{}/panda_rightfinger".format(agent_idx+1),
                    "joint_prim_names": config["robot_info"]["joint_prim_names"],
                    "robot_path": config["robot_info"]["robot_path"],
                    "translation_offset": np.array([0,0+agent_idx*2,0]),
                    "end_effector_prim_name": config["robot_info"]["end_effector_prim_name"]}
        if task_name == "stacking":
            # TODO: Object order length -> Should be at least two objects to show stacking 
            task.add_robot(robot_info=robot_info, object_order_lst=object_order_lst)
        else: task.add_robot(robot_info=robot_info)
        total_object_lst.append(object_order_lst)
    # Main function
    task.main(epochs=2)

""" Pick-and-Place, Following task
if __name__=="__main__": 
    # YAML
    task_info_dir = Path.joinpath(Path.cwd(), 'omniverse_usage/script/cfg/','pnp_task.yaml')
    with open(task_info_dir, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    # Task environment  
    task_name = 'stacking'
    task = FrankSim(task_name=task_name,plane_name='grid_default')
    # Set Plane
    task.add_plane()
    # Robot, Object, Camera Setup
    for agent_idx in range(config["agent_num"]):
        # Robot
        robot_info = {"name": config["robot_info"]["name"]+"{}".format(agent_idx+1),
                    "prim_path": config["robot_info"]["prim_path"]+"{}".format(agent_idx+1),
                    "end_effector_prim_path": "/World/Franka{}/panda_rightfinger".format(agent_idx+1),
                    "joint_prim_names": config["robot_info"]["joint_prim_names"],
                    "robot_path": config["robot_info"]["robot_path"],
                    "translation_offset": np.array([0,0+agent_idx*2,0]),
                    "end_effector_prim_name": config["robot_info"]["end_effector_prim_name"]}
        if task_name == "stacking":
            # TODO: Object order length -> Should be at least two objects to show stacking 
            task.add_robot(robot_info=robot_info, object_order_lst=["object_{}_1".format(agent_idx+1)])
        else: task.add_robot(robot_info=robot_info)
        # Camera 
        camera_info = {"prim_path": config["camera_info"]["prim_path"]+"{}".format(agent_idx+1),
                       "position": np.array([1.8,0+agent_idx*config["place_offset"],1.4]),
                       "rotation": np.array(config["camera_info"]["rotation"]),
                       "resolution": config["camera_info"]["resolution"]}
        task.add_camera(camera_info=camera_info)
        # Object 
        for object_idx in range(config["object_num"]):
            object_info = {"name": config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1), 
                           "prim_path": config["object_info"]["prim_path"]+"_{}_{}".format(agent_idx+1, object_idx+1),
                           "position": np.array([0.5+object_idx*0.1,0+agent_idx*config["place_offset"],0.3]),
                           "scale": np.array(config["object_info"]["scale"]),
                           "type": config["object_info"]["type"]}
            task.add_object(object_info=object_info)
    # Main function
    task.main()
  """