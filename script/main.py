import sys 
sys.path.append('..')
from env.franka_sim import FrankaSim
from pathlib import Path 
import numpy as np
import argparse
import yaml

def main(args):
    # Load YAML
    if args.task_name == 'pick-and-place': yaml_file = 'pnp_task.yaml'
    elif args.task_name == "stacking": yaml_file = "stack_task.yaml"
    elif args.task_name =="following": yaml_file = "follow_task.yaml"
    task_info_dir = Path.joinpath(Path.cwd(), 'omniverse_usage/script/cfg/', yaml_file)
    with open(task_info_dir, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    # Instance environment  
    task = FrankaSim(task_name=args.task_name, config=config)
    # Main function
    task.main()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--task_name', type=str, default='stacking', help="pick-and-place, following, stacking")
    args    = parser.parse_args()
    main(args)