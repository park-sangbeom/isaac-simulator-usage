from omni.isaac.mjcf import _mjcf
mjcf_interface = _mjcf.acquire_mjcf_interface()

# setup config params
import_config = _mjcf.ImportConfig()
import_config.fix_base = True
mjcf_path = './asset/ambidex/desk/mujoco_model_isaac.xml'
prim_path: "/World/ambidex"
# parse and import file
mjcf_interface.create_asset_mjcf(mjcf_path, prim_path, import_config)