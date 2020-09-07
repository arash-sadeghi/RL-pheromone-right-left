from epuckfunc import *
# np.set_printoptions(threshold=sys.maxsize)
go()
stop()
tx=robot.getTime()
# online is 1. offline is 0
agent=Agent()
# action codes: {0: stop, 1: clockwise (turn to right), 2: counter clockwise (turn to left) , 3 :move forward}
agent.getState()
x=0
t=robot.getTime()
while robot.step(timestep) != -1:
    agent.decide()
    agent.act()
    agent.getNextState()
    agent.calculateReward()
    agent.updateTable()
    agent.reset()
    if robot.getTime()-t>60*15:#60*30:
        agent.saveData()
        t=robot.getTime()
