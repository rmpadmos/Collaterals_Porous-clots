#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Usage:
  resize_mesh.py <mesh_path> <scaling>
  resize_mesh.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from Blood_Flow_1D import GeneralFunctions, docopt

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    file_msh = arguments["<mesh_path>"]
    scaling = float(arguments["<scaling>"])
    # file_msh = "./clustered_mesh.msh"
    msh = GeneralFunctions.MSHfile()
    msh.Loadfile(file_msh)

    # scaling = 1.1
    for node in msh.Nodes:
        node[1] *= scaling
        node[2] *= scaling
        node[3] *= scaling
        # node[1] += -50
        # node[2] += -50
        # node[3] += -50

    new_name = '/'.join(file_msh.split("/")[:-1])+"/resized_mesh.msh"
    msh.Writefile(new_name)
