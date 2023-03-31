from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core import SimulationContext, World
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async
import gc
from abc import abstractmethod
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.manipulators import SingleManipulator 
from omni.isaac.franka.tasks import Stacking
from omni.isaac.franka.controllers import StackingController
from omni.isaac.manipulators.grippers import ParallelGripper
from typing import Optional
import sys 
sys.path.append('..')
from env.omni_base import OmniBase
import math 

class FrankTask(OmniBase):
    def __init__(self, name: str,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_cube = None
        self._cube = None
        self._cube_initial_position = cube_initial_position
        self._cube_initial_orientation = cube_initial_orientation
        self._target_position = target_position
        self._cube_size = cube_size
        if self._cube_size is None:
            self._cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if self._cube_initial_position is None:
            self._cube_initial_position = np.array([0.3, 0.3, 0.3]) / get_stage_units()
        if self._cube_initial_orientation is None:
            self._cube_initial_orientation = np.array([1, 0, 0, 0])
        if self._target_position is None:
            self._target_position = np.array([0.6, 0.4, 0]) / get_stage_units()
            self._target_position[2] = self._cube_size[2] / 2.0
        self._target_position = self._target_position + self._offset
        self.env = World()
        self.set_up_scene()

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self):

        robot_path = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        asset_path = get_assets_root_path() + robot_path
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
        gripper = ParallelGripper(
            end_effector_prim_path="/World/Franka/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.05, 0.05]),
        )
        self.env.scene.add_default_ground_plane()
        self._cube = self.env.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="object",
                position=np.array([0.5, 0, 0.1]),
                =self._cube_size,
                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                linear_velocity=np.array([0, 0, 0.01])))
        self._cube2 = self.env.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube2",
                name="object2",
                position=np.array([0.5, 0.2, 0.1]),
                scale=self._cube_size,
                color=np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]),
                linear_velocity=np.array([0, 0, 0.01])))
        
        self._robot = self.env.scene.add(
            SingleManipulator(prim_path="/World/Franka", name="{}".format("Franka"), end_effector_prim_name='panda_rightfinger', gripper=gripper)
        )
        self.object_lst = ["object", "object2"]

    async def setup_pre_reset(self):
            if self.env.physics_callback_exists("sim_step"):
                self.env.remove_physics_callback("sim_step")
            self._controller.reset()
            return
    
    def world_cleanup(self):
        self._controller = None
        return
    
    async def _on_stacking_event_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_stacking_physics_step)
        await world.play_async()
        return

    def _on_stacking_physics_step(self):
        my_franka = self.env.scene.get_object("Franka")
        my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
        self._controller = StackingController(
            name="stacking_controller",
            gripper=my_franka.gripper,
            robot_articulation=my_franka,
            picking_order_cube_names=self.object_lst,
            robot_observation_name=my_franka.name,
        )
        self._articulation_controller = my_franka.get_articulation_controller()
        return
    

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self) -> dict:
        observations = dict()
        print("self._robot.get_joints_state()", self._robot.get_joints_state())
        for name in self.object_lst:
            obj = self.env.scene.get_object(name=name)
            observations.update({name: {"position":obj.get_local_pose()[0], "orientation":obj.get_local_pose()[1], "target_position":self._target_position}})
        observations.update({self._robot.name: {"joint_positions":self._robot.get_joints_state().positions,"end_effector_position":self._robot.end_effector.get_local_pose()[0]}})
        return observations
    
    def get_params(self) -> dict:
        params_representation = dict()
        position, orientation = self._cube.get_local_pose()
        params_representation["cube_position"] = {"value": position, "modifiable": True}
        params_representation["cube_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["target_position"] = {"value": self._target_position, "modifiable": True}
        params_representation["cube_name"] = {"value": self._cube.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation
    
    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        return
    
    def world_cleanup(self):
        self._controller = None
        return
    
    def main(self):
        # Simulation Init 
        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        self._on_stacking_physics_step()
        while simulation_app.is_running():
            self.env.step(render=True)
            obs = self.get_observations()
            actions = self._controller.forward(observations=obs)
            self._articulation_controller.apply_action(actions)
            if self._controller.is_done():
                print("Controller is done")
                self.env.pause()
        simulation_app.close()

if __name__=="__main__":
    franka = FrankTask(name="Franka", cube_size=np.array([0.0515, 0.0515, 0.0515]))
    franka.main()