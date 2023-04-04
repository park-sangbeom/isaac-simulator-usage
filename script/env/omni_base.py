from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async
from omni.isaac.core.simulation_context import SimulationContext
import gc
from typing import Optional
from abc import abstractmethod
import numpy as np 

class OmniBase(object):
    def __init__(self, name: str, offset: Optional[np.ndarray] = None) -> None:
        self._world = None
        self._current_tasks = None
        self._world_settings = {"physics_dt": 1.0 / 60.0, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / 60.0}
        self._scene = None
        self._name = name
        self._offset = offset
        self._task_objects = dict()
        if self._offset is None:
            self._offset = np.array([0.0, 0.0, 0.0])

        if SimulationContext.instance() is not None:
            self._device = SimulationContext.instance().device
        return

    @property
    def device(self):
        return self._device

    @property
    def scene(self) -> Scene:
        """Scene of the world

        Returns:
            Scene: [description]
        """
        return self._scene

    @property
    def name(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return self._name
    
    def get_world(self):
        return self._world

    def set_up_scene(self, scene: Scene) -> None:
        """Adding assets to the stage as well as adding the encapsulated objects such as XFormPrim..etc
           to the task_objects happens here.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        return
    
    def set_world_settings(self, physics_dt=None, stage_units_in_meters=None, rendering_dt=None):
        if physics_dt is not None:
            self._world_settings["physics_dt"] = physics_dt
        if stage_units_in_meters is not None:
            self._world_settings["stage_units_in_meters"] = stage_units_in_meters
        if rendering_dt is not None:
            self._world_settings["rendering_dt"] = rendering_dt
        return

    async def load_world_async(self):
        """Function called when clicking load buttton
        """
        if World.instance() is None:
            await create_new_stage_async()
            self._world = World(**self._world_settings)
            await self._world.initialize_simulation_context_async()
            self.setup_scene()
        else:
            self._world = World.instance()
        self._current_tasks = self._world.get_current_tasks()
        await self._world.reset_async()
        await self._world.pause_async()
        await self.setup_post_load()
        if len(self._current_tasks) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)
        return

    async def reset_async(self):
        """Function called when clicking reset buttton
        """
        if self._world.is_tasks_scene_built() and len(self._current_tasks) > 0:
            self._world.remove_physics_callback("tasks_step")
        await self._world.play_async()
        await update_stage_async()
        await self.setup_pre_reset()
        await self._world.reset_async()
        await self._world.pause_async()
        await self.setup_post_reset()
        if self._world.is_tasks_scene_built() and len(self._current_tasks) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)
        return

    @abstractmethod
    async def setup_post_load(self):
        """called after first reset of the world when pressing load, 
            intializing provate variables happen here.
        """
        return

    @abstractmethod
    async def setup_pre_reset(self):
        """ called in reset button before resetting the world
         to remove a physics callback for instance or a controller reset
        """
        return

    @abstractmethod
    async def setup_post_reset(self):
        """ called in reset button after resetting the world which includes one step with rendering
        """
        return

    @abstractmethod
    async def setup_post_clear(self):
        """called after clicking clear button 
          or after creating a new stage and clearing the instance of the world with its callbacks
        """
        return

    # def log_info(self, info):
    #     self._logging_info += str(info) + "\n"
    #     return

    def _world_cleanup(self):
        self._world.stop()
        self._world.clear_all_callbacks()
        self._current_tasks = None
        self.world_cleanup()
        return

    def world_cleanup(self):
        """Function called when extension shutdowns and starts again, (hot reloading feature)
        """
        return

    async def clear_async(self):
        """Function called when clicking clear buttton
        """
        await create_new_stage_async()
        if self._world is not None:
            self._world_cleanup()
            self._world.clear_instance()
            self._world = None
            gc.collect()
        await self.setup_post_clear()
        return


    def get_observations(self) -> dict:
        """Returns current observations from the objects needed for the behavioral layer.

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Returns True of the task is done.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def cleanup(self) -> None:
        """Called before calling a reset() on the world to removed temporarly objects that were added during
           simulation for instance.
        """
        return

    def set_params(self, *args, **kwargs) -> None:
        """Changes the modifiable paramateres of the task

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Gets the parameters of the task.
           This is defined differently for each task in order to access the task's objects and values.
           Note that this is different from get_observations. 
           Things like the robot name, block name..etc can be defined here for faster retrieval. 
           should have the form of params_representation["param_name"] = {"value": param_value, "modifiable": bool}
    
        Raises:
            NotImplementedError: [description]

        Returns:
            dict: defined parameters of the task.
        """
        raise NotImplementedError