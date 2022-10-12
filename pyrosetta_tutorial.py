#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:02:02 2019

@author: mohseni
"""

#pyrosetta tutorial!!

from pyrosetta import *
from pyrosetta.toolbox import cleanATOM
from pyrosetta.teaching import *
from pyrosetta.rosetta.utility import * 
init()

scorefxn = get_fa_scorefxn()

cleanATOM("1IAR.pdb")
start_pose = pose_from_pdb("1IAR.clean.pdb")

print(str(start_pose.aa(10)))

from pyrosetta.rosetta.protocols.scoring import Interface
from pyrosetta.teaching import *


jump_num = 1
setup_foldtree(start_pose, "A_B", Vector1([jump_num]))
print(start_pose.fold_tree()) 
print(start_pose.jump(jump_num).get_rotation())
print(start_pose.jump(jump_num).get_translation())

myinterface = Interface(jump_num)
myinterface.distance(1000.0)
myinterface.calculate(start_pose) 
a = list(myinterface.pair_list())
for i in a:
    print(list(i))
myinterface.print(start_pose)
start_pose.aa(1147)
start_pose.chain(1147)

from pyrosetta.toolbox import generate_resfile_from_pose
generate_resfile_from_pose(start_pose, "1IAR.resfile")


test = Pose()
test.assign(start_pose)

start_pose.pdb_info().name("start")
test.pdb_info().name("test")

from pyrosetta.teaching import PyMOLMover
pmm = PyMOLMover()
pmm.apply(start_pose)
pmm.apply(test)
pmm.keep_history(True)


from rosetta.protocols.simple_moves import *
min_mover = MinMover()
mm4060 = MoveMap()
mm4060.set_bb(myinterface,True)
movemap.set_chi(myinterface,True)
min_mover.movemap(mm4060)
min_mover.score_function(scorefxn)
min_mover.apply(test)
test.dump_pdb("test.pdb")