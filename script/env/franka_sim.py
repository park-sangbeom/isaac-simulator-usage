from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core import SimulationContext,World
from omni.isaac.core.objects import cuboid, sphere, capsule, cylinder, cone, ground_plane
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.manipulators import SingleManipulator 
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.franka.controllers import StackingController
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.sensor import Camera
from omni.isaac.mjcf import _mjcf

from env.omni_base import OmniBase
from typing import Optional, Union, Dict, List
import matplotlib.pyplot as plt 
from pathlib import Path 
from PIL import Image 
import numpy as np 
import yaml 
import carb 

class FrankaSim(OmniBase):
    def __init__(self, task_name: str = "pick-and-place", 
                       config: Optional[dict] = None):
        self.robot_lst          = []
        self.controller_lst     = []
        self.articulation_lst   = []
        self.object_lst         = []
        self.camera_lst         = []
        self.env                = World()
        self.simulation_context = SimulationContext()
        self.config             = config 
        self.task_name          = task_name 
        self.plane_name         = self.config["plane_name"]
        self.max_count          = self.config["agent_num"]
        np.random.seed(self.config["seed"])
        self.set_up_scene()

    def init_simulation(self):
        self.simulation_context.initialize_physics()
        self.simulation_context.play()

    def init_world(self): 
        self.env.pause()
        self.env.reset()
        self.reset_object()
        self.reset_controller()

    def reset_object(self): 
        for obj_name in self.object_lst:
            self.env.scene.remove_object(name=obj_name)

    def reset_controller(self):
        self.controller_lst = []

    def set_up_scene(self):
        # Plane
        self.add_plane()
        for agent_idx in range(self.config["agent_num"]):
            # Camera 
            camera_info = {"prim_path": self.config["camera_info"]["prim_path"]+"{}".format(agent_idx+1),
                        "position": np.array([1.8,0+agent_idx*self.config["place_offset"],1.4]),
                        "rotation": np.array(self.config["camera_info"]["rotation"]),
                        "resolution": self.config["camera_info"]["resolution"]}
            self.add_camera(camera_info=camera_info)
            # Robot
            robot_info = {"name": self.config["robot_info"]["name"]+"{}".format(agent_idx+1),
                        "prim_path": self.config["robot_info"]["prim_path"]+"{}".format(agent_idx+1),
                        "end_effector_prim_path": "/World/Franka{}/panda_rightfinger".format(agent_idx+1),
                        "joint_prim_names": self.config["robot_info"]["joint_prim_names"],
                        "robot_path": self.config["robot_info"]["robot_path"],
                        "translation_offset": np.array([0,0+agent_idx*2,0]),
                        "end_effector_prim_name": self.config["robot_info"]["end_effector_prim_name"]}
            self.add_robot(robot_info=robot_info)

    # TODO: Get parameters from envrionment
    def get_params(self):
        pass 

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

    def add_robot(self, robot_info: Dict[str,str]):
        robot_path = robot_info["robot_path"]
        asset_path = get_assets_root_path() + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path=robot_info["prim_path"])
        gripper=ParallelGripper(
            end_effector_prim_path=robot_info["end_effector_prim_path"],
            joint_prim_names=robot_info["joint_prim_names"],
            joint_opened_positions=np.array([0.05, 0.05])/ get_stage_units(),
            joint_closed_positions=np.array([0, 0]), # np.array([0.0, 0.0])
            action_deltas=np.array([0.05, 0.05])/ get_stage_units())
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
            self.controller_lst.append(self.controller)
            # Articulation controller
            self.articulation_controller = robot.get_articulation_controller()
            self.articulation_lst.append(self.articulation_controller)

    def add_stacking_controller(self, robot_name: str, object_order_lst: List): 
        robot = self.env.scene.get_object(robot_name)
        # Controller Setup
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

    def add_object(self, object_info: Dict[str,str]):
        if object_info["type"] == "fixed cuboid":
            self.env.scene.add(cuboid.FixedCuboid( 
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.random.uniform(size=(3,))
                                ))
        
        elif object_info["type"]=="dynamic cuboid":
            # TODO: Figure out which parameters are fit for manipulations 
            static_friction = 0.2
            dynamic_friction = 0.1
            restitution = 0.1
            physics_material_path = find_unique_string_name(
                initial_name="/World/Physics_Materials/physics_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            physics_material = PhysicsMaterial(
                prim_path=physics_material_path,
                dynamic_friction=dynamic_friction,
                static_friction=static_friction,
                restitution=restitution,
            )
            self.env.scene.add(cuboid.DynamicCuboid(
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.random.uniform(size=(3,)),
                                mass= 0.001,
                                linear_velocity=np.array([0,0,0]),
                                angular_velocity=np.array([0,0,0]), 
                                physics_material=physics_material
                                ))
        elif object_info["type"]=="visual cuboid": 
            self.env.scene.add(cuboid.VisualCuboid(
                                name=object_info["name"],
                                prim_path=object_info["prim_path"],
                                position=object_info["position"],
                                scale=object_info["scale"],
                                color=np.random.uniform(size=(3,))
                                ))
        self.object_lst.append(object_info["name"])

    def add_camera(self, camera_info: Dict[str,str]):
        self.camera = Camera(
            prim_path=camera_info["prim_path"],
            position=camera_info["position"],
            frequency=20,
            resolution=camera_info["resolution"],
            orientation=rot_utils.euler_angles_to_quats(camera_info["rotation"], degrees=True))
        self.camera_lst.append(self.camera)

    def get_rgb(self, rel_dir: str = "omniverse_usage/script/data/npy/", 
                      image_name: str = "test.npy", 
                      SAVE: bool = False) -> List:
        rgb_lst = []
        for camera_idx, camera in enumerate(self.camera_lst):
            rgb = camera.get_rgba()[:,:,:3]
            rgb_lst.append(rgb)
            if SAVE: 
                path = Path.joinpath(Path.cwd(), rel_dir, str(camera_idx)+image_name)
                np.save(path, rgb)
        return rgb_lst 
    
    def get_proprioception(self) -> dict:
        self.proprioception = dict()
        # Object observation
        if self.task_name=="stacking":
            for obj_idx, (obj_name, target_position) in enumerate(zip(self.object_lst, self.target_position_lst)):
                i = obj_idx%self.config["object_num"]
                obj = self.env.scene.get_object(name=obj_name)
                self.proprioception.update({obj_name: {"position":obj.get_local_pose()[0], 
                                                       "orientation":obj.get_local_pose()[1], 
                                                       "target_position":np.array(
                                                        [target_position[0],
                                                         target_position[1],
                                                        (self.config["object_info"]["scale"][2] * i) + self.config["object_info"]["scale"][2]/ 2.0])
                                                        }})                                  
        else:
            for obj_name in self.object_lst:
                obj = self.env.scene.get_object(name=obj_name)
                self.proprioception.update({obj_name: {"position":obj.get_local_pose()[0], "orientation":obj.get_local_pose()[1]}})
        # Robot observation
        for robot_name in self.robot_lst:
            robo = self.env.scene.get_object(name=robot_name)
            self.proprioception.update({robot_name: {"joint_positions":robo.get_joints_state().positions,"end_effector_position":robo.end_effector.get_local_pose()[0]}})
        return self.proprioception
 

    def stack(self, epochs: int=1): 
        """
        Stakcing objects Task
        """
        self.init_simulation()

        for _ in range(epochs):
            self.target_position_lst=[]
            reset_count=0 
            # Random Initialization
            for agent_idx in range(self.config["agent_num"]):
                robot_name = "Franka{}".format(agent_idx+1)
                object_order_lst=[];
                # Set target position 
                target_position = np.array([np.random.uniform(0.5,0.7), 0.2+agent_idx*2.0, 0.0515])
                for object_idx in range(self.config["object_num"]):
                    init_position = (np.array([np.random.uniform(0.5,0.7), -0.2+agent_idx*self.config["place_offset"], 0.06]))
                    object_info = {"name": self.config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1), 
                                "prim_path": self.config["object_info"]["prim_path"]+"_{}_{}".format(agent_idx+1, object_idx+1),
                                "position": init_position,
                                "scale": np.array(self.config["object_info"]["scale"]),
                                "type": self.config["object_info"]["type"],
                                "target_position": target_position}
                    self.target_position_lst.append(target_position)
                    self.add_object(object_info=object_info)
                    object_order_lst.append(self.config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1))
                self.add_stacking_controller(robot_name=robot_name, object_order_lst=object_order_lst)

            while simulation_app.is_running():
                self.env.step(render=True)
                obs = self.get_proprioception()
                for controller, articulation_controller in zip(self.controller_lst, self.articulation_lst):
                    actions = controller.forward(obs)
                    articulation_controller.apply_action(actions)
                    if controller.is_done():
                        print("Controller is done")
                        reset_count+=1 
                if reset_count == self.max_count:
                    self.init_world()
                    reset_count=0
                    break 
        simulation_app.close()
   
    def follow_target(self, epochs: int=1):
        """
            Following a target object Task
        """
        self.init_simulation()

        for _ in range(epochs):
            reset_count=0 
            self.add_controller()
            # Random Initialization
            for agent_idx in range(self.config["agent_num"]):
                for object_idx in range(self.config["object_num"]):
                    random_position = (np.array([np.random.uniform(0.5,0.7), np.random.uniform(-0.3,0.3)+agent_idx*self.config["place_offset"], np.random.uniform(0.2,0.6)]))
                    object_info = {"name": self.config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1), 
                                "prim_path": self.config["object_info"]["prim_path"]+"_{}_{}".format(agent_idx+1, object_idx+1),
                                "position": random_position,
                                "scale": np.array(self.config["object_info"]["scale"]),
                                "type": self.config["object_info"]["type"]}
                    self.add_object(object_info=object_info)
            
            while simulation_app.is_running():
                self.env.step(render=True)
                obs = self.get_proprioception()
                for agent_idx, (robot_name, controller, articulation_controller) in enumerate(zip(self.robot_lst, self.controller_lst, self.articulation_lst)):
                    my_franka = self.env.scene.get_object(robot_name)
                    end_effector_position, _  = my_franka.gripper.get_world_pose()
                    target_name="object_{}_1".format(agent_idx+1)
                    actions = controller.forward(
                        target_end_effector_position=obs[target_name]["position"],
                        target_end_effector_orientation=obs[target_name]["orientation"])
                    articulation_controller.apply_action(actions)
                    if np.mean(np.abs(np.array(end_effector_position)-np.array(obs[target_name]["position"]))<(0.01 / get_stage_units())):
                        print("Controller is done")
                        reset_count+=1 
                if reset_count == self.max_count:
                    self.init_world()
                    reset_count=0
                    break 
        simulation_app.close()

    def pick_and_place(self, epochs: int=1): 
        """
            Pick and Place Task
        """
        self.init_simulation()

        for _ in range(epochs):
            reset_count=0 
            # Set target position
            place_position_lst = []
            for _agent_idx in range(len(self.robot_lst)): 
                place_position_lst.append(np.array([np.random.uniform(0.5,0.7), np.random.uniform(-0.3,0.3)+_agent_idx*self.config["place_offset"], 0.0515]))
            self.add_controller()
            # Random Initialization
            for agent_idx in range(self.config["agent_num"]):
                for object_idx in range(self.config["object_num"]):
                    random_position = (np.array([np.random.uniform(0.5,0.7), np.random.uniform(-0.3,0.3)+agent_idx*self.config["place_offset"], self.config["object_info"]["scale"][2]/2.0]))
                    object_info = {"name": self.config["object_info"]["name"]+"_{}_{}".format(agent_idx+1, object_idx+1), 
                                "prim_path": self.config["object_info"]["prim_path"]+"_{}_{}".format(agent_idx+1, object_idx+1),
                                "position": random_position,
                                "scale": np.array(self.config["object_info"]["scale"]),
                                "type": self.config["object_info"]["type"]}
                    self.add_object(object_info=object_info)

            while simulation_app.is_running():
                self.env.step(render=True)
                obs = self.get_proprioception()
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
                    self.init_world()
                    reset_count=0
                    break 
        simulation_app.close()

    def main(self): 
        if self.task_name=="pick-and-place":
            self.pick_and_place(epochs=self.config["epochs"])
        elif self.task_name=="following":
            self.follow_target(epochs=self.config["epochs"])
        elif self.task_name=="stacking":
            self.stack(epochs=self.config["epochs"])


if __name__=="__main__": 
    task_name = 'following'
    # Load YAML
    if task_name == 'pick-and-place': yaml_file = 'pnp_task.yaml'
    elif task_name == "stacking": yaml_file = "stack_task.yaml"
    elif task_name =="following": yaml_file = "follow_task.yaml"
    task_info_dir = Path.joinpath(Path.cwd(), 'cfg/', yaml_file)
    with open(task_info_dir, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    # Instance environment  
    task = FrankaSim(task_name=task_name, config=config)
    # Main function
    task.main()