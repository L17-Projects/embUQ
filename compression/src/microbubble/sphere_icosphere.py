#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

import trimesh
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--simnum", dest="simnum", default="00001")
parser.add_argument("-o", dest="fname", type=str, default="emb.off")
parser.add_argument("-s", dest="subDiv", type=int, default=4)
parser.add_argument("-r", dest="radp", type=float, default=4.0)
args = parser.parse_args()

filename_default = "../parameter/parameters-default" + args.simnum + ".yaml"
with open(filename_default, "rb") as f:
    parameters_default = yaml.load(f, Loader=yaml.CLoader)

subDiv = args.subDiv
fname = args.fname
radp = args.radp
m = trimesh.creation.icosphere(subdivisions=subDiv, radius=radp)
a = m.export(fname, "off")
