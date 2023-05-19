from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import SimulationContext,World
from omni.isaac.core.objects import cuboid, sphere, capsule, cylinder, cone, ground_plane
from omni.isaac.core.utils.stage import add_reference_to_stage,get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.sensor import Camera
from env.omni_base import OmniBase
from typing import Optional, Union, Dict, List
import matplotlib.pyplot as plt 
from pathlib import Path 
from PIL import Image 
import numpy as np 
import cv2
import carb 
from PIL import Image
from script.model.scene_detection import ovd_inference
from script.model.gpt.chat_gpt import ChatGPT
from script.model.llava.llava_api import LLaVA 
import time 

class OmniverseEnvironment(OmniBase):
    def __init__(self, task_name: str = "sample_scene", 
                       plane_name: str="grid_default",
                       camera_prim_path: str="/World/Camera",
                       camera_position: List=[1.8, 0,1.4],
                       camera_rotation: List=[180,135,0],
                       camera_resolution: List=[256, 256],
                       model: str="llava",
                       random_seed: int=42
                       ):
        self.robot_lst          = []
        self.object_lst         = []
        self.env                = World(stage_units_in_meters=1.0)
        self.simulation_context = SimulationContext()
        self.task_name          = task_name 
        self.plane_name         = plane_name
        self.camera_prim_path   = camera_prim_path
        self.camera_position    = camera_position
        self.camera_rotation    = camera_rotation
        self.camera_resolution  = camera_resolution
        self.model              = model
        np.random.seed(random_seed)
        self.init_simulation()
        print("Done.")

    def init_simulation(self):
        self.simulation_context.initialize_physics()
        self.simulation_context.play()
        print("Model: {}".format(self.model))
        if self.model == 'gpt-3.5-turbo':
            self.llm_model = ChatGPT(model_engine=self.model)
        elif self.model ==  'text-davinci-003':
            self.llm_model = ChatGPT(model_engine=self.model)
        elif self.model=='llava':
            self.llm_model = LLaVA()

    def init_world(self): 
        self.env.pause()
        self.env.reset()
        self.reset_object()


    def number_demo(self):
        object_info = {"type":"fixed cuboid",
                        "prim_path":"/World/table",
                        "position":np.array([0,0,0.5]),
                        "scale":[1, 1, 1],
                        "name":"table"}
        self.add_object(object_info=object_info)
        # self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BooksSet_25.usd", name="bookset_25", position=np.array([0.3, -0.35,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleA.usd", name="bottle_a", position=np.array([0, 0.1,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleB.usd",name="bottle_b", position=np.array([0.2, -0.15,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleC.usd", name="bottle_c", position=np.array([0.15, 0.05,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleD.usd", name="bottle_d", position=np.array([-0.2, -0.1,1.2]))
        first_q = True  
        second_q = True 
        third_q = True 
        fourth_q = True 

        i=0
        while simulation_app.is_running():
            self.env.step(render=True)
            i+=1
            #1 Q 
            if first_q==True and i>100:
                self.llm_model.refresh()
                image_name = 'sample_scene1'
                rgb = self.get_rgb(image_name=image_name)
                prompt = "I would like to give a bottle each of them, How many bottles are there?"
                print("Prompt: {}".format(prompt))
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('./data/{}.png'.format(image_name))
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                first_q=False 
                self.move_object(name='bottle_a', position=np.array([0, -3,1.2]))
        
            #2 Q 
            if second_q==True and i>300:
                self.llm_model.refresh()
                image_name = 'sample_scene2'
                rgb = self.get_rgb(image_name=image_name)
                prompt = "How many bottles are there?"
                print("Prompt: {}".format(prompt))
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('./data/{}.png'.format(image_name))
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                second_q=False 
                self.move_object(name='bottle_b', position=np.array([0, -3,1.2]))

            #3 Q 
            if third_q==True and i>500:
                self.llm_model.refresh()
                image_name = 'sample_scene3'
                rgb = self.get_rgb(image_name=image_name)
                prompt = "How many bottles are there?"
                print("Prompt: {}".format(prompt))
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('./data/{}.png'.format(image_name))
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                third_q=False 
                self.move_object(name='bottle_c', position=np.array([0, -3,1.2]))
                self.move_object(name='bottle_d', position=np.array([0, -3,1.2]))
            #4 Q 
            if fourth_q==True and i>1000:
                self.llm_model.refresh()
                image_name = 'sample_scene4'
                rgb = self.get_rgb(image_name=image_name)
                prompt = "How many bottles are there?"
                print("Prompt: {}".format(prompt))
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('./data/{}.png'.format(image_name))
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                fourth_q=False 
        simulation_app.close()

    def direction_demo(self):
        object_info = {"type":"fixed cuboid"
                        "prim_path":"/World/table",
                        "position":np.array([0,0,0.5]),
                        "scale":[1, 1, 1],
                        "name":"table"}
        self.add_object(object_info=object_info)
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BooksSet_25.usd", name="bookset_25", position=np.array([0.3, -0.35,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleA.usd", name="bottle_a", position=np.array([0, 0.2,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleB.usd",name="bottle_b", position=np.array([0.2, 0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleC.usd", name="bottle_c", position=np.array([0.15, 0.35,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleD.usd", name="bottle_d", position=np.array([-0.2, 0.4,1.2]))
        first_q = True  
        second_q = True 
        move_flag = True 
        i=0
        while simulation_app.is_running():
            self.env.step(render=True)
            i+=1
            # First Q 
            if first_q==True and i>100:
                self.llm_model.refresh()
                image_name = 'sample_scene1'
                rgb = self.get_rgb(image_name=image_name)
                prompt = "Is the red book on the left side or right side?"
                # prompt = "Is the red book on the left side or right side? And are bootles on the left side or right side?"
                print("Prompt: {}".format(prompt))
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('./data/{}.png'.format(image_name))
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                first_q=False 
            else: pass 
        
            # Move 
            if move_flag==True and i>100:
                self.move_object(name='bookset_25', position=np.array([0, 0.3,1.2]))
                self.move_object(name='bottle_a', position=np.array([0, -0.3,1.2]))
                self.move_object(name='bottle_b', position=np.array([-0.2, -0.3,1.2]))
                self.move_object(name='bottle_c', position=np.array([-0.1, -0.4,1.2]))
                self.move_object(name='bottle_d', position=np.array([-0.3, -0.35,1.2]))
                move_flag = False 
            else: pass 

            # Second Q 
            if second_q==True and i>200:
                self.llm_model.refresh()
                image_name = 'sample_scene2'
                rgb = self.get_rgb(image_name=image_name)
                prompt = "Is the red book on the left side or right side?"
                # prompt = "Is the red book on the left side or right side? And are bootles on the left side or right side?"
                print("Prompt: {}".format(prompt))
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('./data/{}.png'.format(image_name))
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                second_q=False 
            else: pass 

        simulation_app.close()

    def safety_demo(self):
        object_info = {"type":"fixed cuboid",
                        "prim_path":"/World/table",
                        "position":np.array([0,0,0.5]),
                        "scale":[1, 1, 1],
                        "name":"table"}
        self.add_object(object_info=object_info)
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BooksSet_25.usd", name="bookset_25", position=np.array([0.3, 0.2,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleA.usd", name="bottle_a", position=np.array([0, 0.1,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleB.usd",name="bottle_b", position=np.array([0, 0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleC.usd", name="bottle_c", position=np.array([0, -0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleD.usd", name="bottle_d", position=np.array([-0.2, 0.15,1.2]))
        first_q = True  
        while simulation_app.is_running():
            self.env.step(render=True)
            if first_q==True: 
                prompt = "There are bunch of objects in my desk\
                        I would like to do something in my room but want to just make sure that \
                        wheter there is a dangerous object or not Let me know the name of object \
                        if there is a dangerous object in my desk. Then, what should I do for the next?"
                if self.model=='gpt-3.5-turbo':
                    reply  = self.llm_model.get_answer(prompt=prompt)
                elif self.model=='text-davinci-003':
                    reply  =self.llm_model.get_answer(prompt=prompt)
                elif self.model=='llava':
                    pil_image = Image.open('color_scene.png')
                    reply = self.llm_model.request(question=prompt, image=pil_image)
                first_q=False 
            else: 
                pass 
        simulation_app.close()

    def scene_demo(self):
        object_info = {"type":"fixed cuboid",
                        "prim_path":"/World/table",
                        "position":np.array([0,0,0.5]),
                        "scale":[1, 1, 1],
                        "name":"table"}
        self.add_object(object_info=object_info)
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BooksSet_25.usd", name="bookset_25", position=np.array([0.3, 0.2,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleA.usd", name="bottle_a", position=np.array([0, 0.1,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleB.usd",name="bottle_b", position=np.array([0, 0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleC.usd", name="bottle_c", position=np.array([0, -0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleD.usd", name="bottle_d", position=np.array([-0.2, 0.15,1.2]))

        count = 0
        while simulation_app.is_running():
                count+=1
                self.env.step(render=True)
                if count==100:
                    self.get_rgb()
        simulation_app.close()

    def random_spawn_demo(self):
        object_info = {"type":"fixed cuboid",
                        "prim_path":"/World/table",
                        "position":np.array([0,0,0.5]),
                        "scale":[1, 1, 1],
                        "name":"table"}
        self.add_object(object_info=object_info)
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BooksSet_25.usd", name="bookset_25", position=np.array([0.3, 0.2,1.2]))
        # self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleA.usd", name="bottle_a", position=np.array([0, 0.1,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleB.usd",name="bottle_b", position=np.array([0, 0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleC.usd", name="bottle_c", position=np.array([0, -0.3,1.2]))
        self.add_asset(usd_path="/Isaac/Environments/Office/Props/SM_BottleD.usd", name="bottle_d", position=np.array([-0.2, 0.15,1.2]))

        count = 0
        while simulation_app.is_running():
                count+=1
                self.env.step(render=True)
                if count%100==0:
                    time.sleep(3)
                    self.move_object(name='bookset_25', position=np.array([np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3),1.1]))
                    # self.move_object(name='bottle_a', position=np.array([0, -0.3,1.2]))
                    self.move_object(name='bottle_b', position=np.array([np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3),1.1]))
                    self.move_object(name='bottle_c', position=np.array([np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3),1.1]))
                    self.move_object(name='bottle_d', position=np.array([np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3),1.1]))
        simulation_app.close()
        


    def reset_object(self): 
        for obj_name in self.object_lst:
            self.env.scene.remove_object(name=obj_name)

    def set_up_scene(self):
        # Plane
        self.add_plane()
        # Camera 
        camera_info = {"prim_path": self.camera_prim_path,
                    "position": np.array(self.camera_position),
                    "rotation": np.array(self.camera_rotation),
                    "resolution": self.camera_resolution}
        self.add_camera(camera_info=camera_info)

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

    def add_asset(self,usd_path="/Isaac/Environments/Grid/default_environment.usd", name='object', position=np.array([1.0, 0.5, 0])):
        prim_path = "/World/{}".format(name)
        asset_path=""
        asset_path = get_assets_root_path()+usd_path
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        # Geometry
        GeometryPrim(prim_path=prim_path,
                     name=name,
                     collision=True,
                     track_contact_forces=True,
                     prepare_contact_sensor=True)
        # Rigid 
        RigidPrim(prim_path=prim_path,
                  name=name,
                  position=position)
        
    def move_object(self, name='object', position=np.array([1.0, 0.5, 0])):
        prim_path = "/World/{}".format(name)
        RigidPrim(prim_path=prim_path,
                  name=name,
                  position=position) 
        
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
            # TODO: Need to figure out the physics parameters for each manipulation tasks 
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

    def add_camera(self,camera_info: Dict[str,str]):
        self.camera = Camera(
            prim_path=camera_info["prim_path"],
            position=camera_info["position"],
            frequency=20,
            resolution=camera_info["resolution"],
            orientation=rot_utils.euler_angles_to_quats(camera_info["rotation"], degrees=True))
        print(rot_utils.euler_angles_to_quats(camera_info["rotation"], degrees=True))

    def get_rgb(self, rel_dir: str = "data/", 
                      image_name: str = "sample_scene", 
                      SAVE: bool = True) -> List:
        rgb = self.camera.get_rgba()[:,:,:3]
        if SAVE: 
            path = Path.joinpath(Path.cwd(), rel_dir, image_name)
            np.save(path, rgb)
            cv2.imwrite(str(path)+'.png', rgb)
        return rgb 
    
    def get_proprioception(self) -> dict:
        self.proprioception = dict()
        # Object observation
        for obj_name in self.object_lst:
            obj = self.env.scene.get_object(name=obj_name)
            self.proprioception.update({obj_name: {"position":obj.get_local_pose()[0], "orientation":obj.get_local_pose()[1]}})
        # Robot observation
        for robot_name in self.robot_lst:
            robo = self.env.scene.get_object(name=robot_name)
            self.proprioception.update({robot_name: {"joint_positions":robo.get_joints_state().positions,"end_effector_position":robo.end_effector.get_local_pose()[0]}})
        return self.proprioception

    # TODO: Get parameters from envrionment
    def get_params(self):
        pass 
