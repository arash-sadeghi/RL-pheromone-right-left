from controller import Supervisor,Display, ImageRef
from time import time,ctime 
from numpy.random import random
import numpy as np
from math import sqrt
import os
ROBN=1
sup=Supervisor()
timestep = int(sup.getBasicTimeStep())
strs=["_"+str(_) for _ in range(1,ROBN+1)]
defs=[sup.getFromDef(_) for _ in strs]
fld=[_.getField("translation") for _ in defs]  
rfld=[_.getField("rotation") for _ in defs]
cst=defs[0].getField("customData")
Time=sup.getTime()
while sup.step(timestep) != -1:
    req=cst.getSFString()
    if req=="reset":
        cst.setSFString('reset done')
        fld[0].setSFVec3f([0,0,-7])       
        rot=[0,1,0,3.14] 
        rfld[0].setSFRotation(rot) 
