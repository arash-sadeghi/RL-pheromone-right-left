from epuckfunc import *
stop()
t=robot.getTime()
while robot.step(timestep) != -1:
    print(stnd_cue())
