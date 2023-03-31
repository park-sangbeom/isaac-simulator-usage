from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async,update_stage_async,get_stage_units,add_reference_to_stage
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.objects import cuboid, sphere, capsule, cylinder, cone, ground_plane
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.manipulators import SingleManipulator 
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from pxr import Gf, UsdGeom
from typing import Optional
import omni 
import carb 
import numpy as np 

class OmniverseEnvironment:
    def __init__(self,map_name='grid_default', headless=False):
        self.world          = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=0.01)
        # self.stage        = omni.usd.get_context().get_stage()
        self.map_name       = map_name
        self.headless       = headless 
        self.camera_lst     = []
        self.object_lst     = []
        self.robot_lst      = []
        self.controller_lst = []
        self.articulation_lst=[]

    def add_env(self):
        asset_path=""
        if self.map_name=="grid_default":
            asset_path = get_assets_root_path()+"/Isaac/Environments/Grid/default_environment.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")

        elif self.map_name=="simple_room":
            asset_path = get_assets_root_path()+"/Isaac/Environments/Simple_Room/simple_room.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")

        elif self.map_name=="office":
            asset_path = get_assets_root_path()+"/Isaac/Environments/Office/office.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        
        else: 
            carb.log_error("Could not find the '{}' environment".format(self.map_name))
    
    def add_object(self, object_info=None):
        if object_info["type"] == "fixed cuboid":
            self.world.scene.add(cuboid.FixedCuboid( 
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)])
                                ))
        
        elif object_info["type"]=="dynamic cuboid":
            self.world.scene.add(cuboid.DynamicCuboid(
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                                ))
        elif object_info["type"]=="visual cuboid": 
            self.world.scene.add(cuboid.VisualCuboid(
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                                ))
        self.object_lst.append(object_info["name"])

    def add_robot(self, robot_info=None):
        robot_path = robot_info["robot_path"]
        asset_path = get_assets_root_path() + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path=robot_info["prim_path"])
        gripper=ParallelGripper(
            end_effector_prim_path=robot_info["end_effector_prim_path"],
            joint_prim_names=robot_info["joint_prim_names"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.05, 0.05]))

        robot=self.world.scene.add((
            SingleManipulator(
            prim_path=robot_info["prim_path"], 
            name=robot_info["name"], 
            end_effector_prim_name=robot_info["end_effector_prim_name"],
            gripper=gripper,
            position=robot_info["translation_offset"])))
        self.robot_lst.append(robot_info["name"])

    def add_controller(self,robot_name):
        robot = self.world.scene.get_object(robot_name)
        robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
        self.controller =  PickPlaceController(name="pick_place_controller", gripper=robot.gripper, robot_articulation=robot)
        self.articulation_controller = robot.get_articulation_controller()
        self.controller_lst.append(self.controller)
        self.articulation_lst.append(self.articulation_controller)

    def add_table(self):
        pass 

    def add_camera(self, prim_path: str, 
                   position: Optional[np.ndarray], 
                   rotation: Optional[np.ndarray], 
                   resolution: tuple=(256,256)):
        self.camera = Camera(
            prim_path=prim_path,
            position=position,
            frequency=20,
            resolution=resolution,
            orientation=rot_utils.euler_angles_to_quats(rotation, degrees=True))
        self.camera_lst.append(self.camera)

    def get_observations(self) -> dict:
        observations = dict()
        for obj_name in self.object_lst:
            obj = self.world.scene.get_object(name=obj_name)
            observations.update({obj_name: {"position":obj.get_local_pose()[0], "orientation":obj.get_local_pose()[1]}})
        for robot_name in self.robot_lst:
            robo = self.world.scene.get_object(name=robot_name)
            observations.update({robot_name: {"joint_positions":robo.get_joints_state().positions,"end_effector_position":robo.end_effector.get_local_pose()[0]}})
        return observations

    def remove_object(self, object_name):
        pass 

    def move_object(self, object_name, offset):
        pass 

    