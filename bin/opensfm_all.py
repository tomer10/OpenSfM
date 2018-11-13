#!/usr/bin/env python

# USAGE EXAMPLE:
#  all /Users/tomerpeled/DB/incident_Lev0518/incident1
# all /Users/tomerpeled/DB/OpenSfM/data/berlin  /Users/tomerpeled/code/OpenSfM/data/lv

# --- view ---
# python -m SimpleHTTPServer
# http://localhost:8000/viewer/reconstruction.html#file=/data/lv/reconstruction.meshed.json
# http://localhost:8000/viewer/reconstruction.html#file=/data/lv_gps/reconstruction.meshed.json

# ToDO:
# add to requirements: matplotlib, pathlib

from os.path import abspath, join, dirname
import sys
sys.path.insert(0, abspath(join(dirname(__file__), "..")))

import argparse
import logging

from opensfm import commands
from opensfm import log


from argparse import Namespace

import sys

# from opensfm.test import data_generation as data_generation
# from opensfm import test.data_generation as data_generation
# from opensfm/test import data_generation as data_generation

# def tomerp_run_all(tmpdir):
#     data = data_generation.create_berlin_test_folder(tmpdir)
#
#     run_all_commands = [
#         commands.extract_metadata,
#         commands.detect_features,
#         commands.match_features,
#         commands.create_tracks,
#         commands.reconstruct,
#         commands.mesh,
#         commands.undistort,
#         commands.compute_depthmaps,
#     ]
#
#     for module in run_all_commands:
#         command = module.Command()
#         run_command(command, [data.data_path])
#
#     reconstruction = data.load_reconstruction()
#     assert len(reconstruction[0].shots) == 3
#     assert len(reconstruction[0].points) > 1000

# print(sys.argv)

log.setup()

# Create the top-level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
    help='Command to run', dest='command', metavar='command')

# Create one subparser for each subcommand
subcommands = [module.Command() for module in commands.opensfm_commands]

for command in subcommands:
    subparser = subparsers.add_parser(
        command.name, help=command.help)
    command.add_arguments(subparser)


# args = parser.parse_args()
# ar_flow=['extract_metadata', 'detect_features', 'match_features', 'create_tracks', 'reconstruct', 'mesh', 'undistort', 'compute_depthmaps']
# args_cur = args  # {"name": 'all', "dataset": sys.argv[2]}
# for command2run in ar_flow:
#     for command in subcommands:
#         if command2run == command.name:
#             args_cur['name']=command2run
#             print(args_cur['name'])
#             command.run(args_cur)
#
#
# a = Namespace(tomer='hameleh')



# from test.test_commands import tomerp_run_all as tomerp_run_all
# test_commands.py
if sys.argv[1]=='all':
    # tomerp_run_all(sys.argv[2])


    # ar_flow=['extract_metadata']
    ar_flow=['extract_metadata', 'detect_features', 'match_features', 'create_tracks', 'reconstruct', 'mesh', 'undistort', 'compute_depthmaps']
    # ar_flow=[ 'reconstruct', 'mesh', 'undistort', 'compute_depthmaps']
    # args_cur = {"name": 'all', "dataset": sys.argv[2]}
    args_cur= Namespace(name='all', dataset=sys.argv[2], interactive=False)
    # args_cur['interactive'] = False
    for command2run in ar_flow:
        for command in subcommands:
            if command2run == command.name:
                args_cur.name=command2run
                print(args_cur.name)
                command.run(args_cur)
else:
    # Parse arguments
    args = parser.parse_args()
    # Run the selected subcommand
    for command in subcommands:
        if args.command == command.name:
            command.run(args)
