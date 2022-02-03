#! /home/rdaneel/anaconda3/lib/python3.8
# -*- coding: UTF-8 -*-
"""Utility functions"""

import logging
import os
import sys
from typing import Dict
import pandas as pd

def read_file(in_path:str) -> pd.DataFrame:
    """ Read the csv file with pandas

    Args:
        in_path (str): path to the csv file

    Returns:
        pd.DataFrame: the csv file
    """
    if not os.path.exists(in_path):
        logging.error('%s does not exist!', in_path)
        sys.exit()
    else:
        return pd.read_csv(in_path)

def get_file_dir(exp_path:str, map_name:str, solver_name:str) -> str:
    """Get the path to the csv files

    Args:
        exp_path (str): path to the whole experiments
        map_name (str): map name
        solver_name (str): solver name

    Returns:
        str: path to the csv files
    """
    map_dir = os.path.join(exp_path, map_name)
    out_dir = os.path.join(map_dir, solver_name)
    return out_dir

def get_solver(solver:Dict)->Dict:
    """get the solver as a dictionary

    Args:
        solver (Dict): solver from config.yaml

    Returns:
        Dict: solver after filtering out the undeclair variables
    """
    def get_solver_by_key(_key: str):
        out_val = solver[_key] if _key in solver.keys() else -1
        return out_val

    out_solver = dict()

    # Basic parameters
    out_solver['name'] = get_solver_by_key('name')
    out_solver['label'] = get_solver_by_key('label')
    out_solver['color'] = get_solver_by_key('color')
    out_solver['marker'] = get_solver_by_key('marker')

    # Bounded suboptimal solvers
    out_solver['w'] = get_solver_by_key('w')
    out_solver['mth'] = get_solver_by_key('mth')

    # FEECBS parameters
    out_solver['T_N'] = get_solver_by_key('T_N')
    out_solver['T_i'] = get_solver_by_key('T_i')

    return out_solver

def get_file_name(map_name:str, scen:str, ag_num:int, solver: Dict) -> str:
    """Get the name of the csv file (end with .csv)

    Args:
        map_name (str): map_name
        scen (str): even or random scen
        ag_num (int): number of agents
        solver (Dict): the solver from config.yaml
    Returns:
        str: name of the csv files
    """
    tmp_solver = get_solver(solver)
    f_w = ('{0:.2f}'.format(tmp_solver['w']) + '-') if tmp_solver['w'] > 1 else ''
    mth = (str(tmp_solver['mth']) + '-') if tmp_solver['mth'] > -1 else ''
    out_name = map_name + '-' + scen + '-' + str(ag_num) + '-' + str(f_w) + str(0)\
        + '-' + mth + tmp_solver['name'] + '.csv'
    return out_name

def get_csv_instance(exp_path:str, map_name:str, scen:str, ag_num:int, solver:Dict):
    """Get the path and read the csv with pandas

    Args:
        map_name (str): map_name
        scen (str): even or random scen
        ag_num (int): number of agents
        solver (Dict): the solver from config.yaml
    Returns:
        pd.DataFrame: the csv file
    """
    tmp_solver = get_solver(solver)
    return read_file(os.path.join(
        get_file_dir(exp_path, map_name, solver['name']),
        get_file_name(map_name, scen, ag_num, tmp_solver)))
