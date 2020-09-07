from controller import Robot,Keyboard,Motor, PositionSensor
import numpy as np
from math import sqrt,atan2,sin,cos,atan
import random
from time import time ,ctime
import itertools as it
from termcolor import colored as c
import cv2 as cv
####################################################################################################################
robot = Robot()
name= robot.getName()
Ftime=10000
FW=4
rot_coeff=0.0236733
arena_x=2
arena_y=4
Lx=2
Ly=2*Lx
SEN_THRESH=470
COL_SEN_THR=200
collision_recognition=" "
flag=False
timestep = int(robot.getBasicTimeStep())
motors=[[],[]]
motors[0]=robot.getMotor("left wheel motor")
motors[1]=robot.getMotor("right wheel motor")
motors[0].setPosition(float("inf"))
motors[1].setPosition(float("inf"))
motors[0].setVelocity(0)
motors[1].setVelocity(0)

ps=[[],[]]
ps[0]=motors[0].getPositionSensor()
ps[0].enable(timestep)
ps[1]=motors[1].getPositionSensor()
ps[1].enable(timestep)

rr=robot.getDistanceSensor('rr')
rr.enable(timestep)
ll=robot.getDistanceSensor('ll')
ll.enable(timestep)

ds=[[np.nan] for _ in range(0,8)]
ds[0]= robot.getDistanceSensor('ps0')
ds[0].enable(timestep)
ds[1]= robot.getDistanceSensor('ps1')
ds[1].enable(timestep)
ds[2]= robot.getDistanceSensor('ps2')
ds[2].enable(timestep)
ds[3]= robot.getDistanceSensor('ps3')
ds[3].enable(timestep)
ds[4]= robot.getDistanceSensor('ps4')
ds[4].enable(timestep)
ds[5]= robot.getDistanceSensor('ps5')
ds[5].enable(timestep)
ds[6]= robot.getDistanceSensor('ps6')
ds[6].enable(timestep)
ds[7]= robot.getDistanceSensor('ps7')
ds[7].enable(timestep)

gps=robot.getGPS("gps")
gps.enable(timestep)

comp=robot.getCompass("compass")
comp.enable(timestep)
L0,L1,L7,L8=robot.getCompass("led0"),robot.getCompass("led1"),robot.getCompass("led7"),robot.getCompass("led8")
grandList=list(it.product([0,1,2,3,4,5,6,7,8,9],repeat=6)) ######### 6 #########
fileRep=open('test.npy', 'wb')
logname=ctime(time()).replace(':','_')+'.npy'
ph=cv.imread('ph.png')
####################################################################################################################
def delay(x):
    L8.set(0)
    if x=='inf':
        while robot.step(timestep) != -1: pass
    t=robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime()-t>x: return 1
####################################################################################################################
def go():
    motors[0].setVelocity(FW)
    motors[1].setVelocity(FW)      
####################################################################################################################
def stop():
    motors[1].setVelocity(0)
    motors[0].setVelocity(0)
####################################################################################################################
def make0_360(x):
    while x>360: x-=360
    while x<0: x+=360
    return x
####################################################################################################################
def cord():
    pos=gps.getValues()
    rotx=np.array(comp.getValues())
    rot=atan2(rotx[2],rotx[0])
    rot=rot*180/np.pi
    rot+=180
    if rot>180: rot=-1*(360-rot)
    return [pos[0],pos[2],rot]
####################################################################################################################
def stnd_cue(binirize=False):
    pos=cord()
    senDist=0.03
    rightSen=[pos[0]+senDist/2*sin(pos[2]*np.pi/180-np.pi/2),pos[1]+senDist/2*cos(pos[2]*np.pi/180-np.pi/2)]
    leftSen=[pos[0]-senDist/2*sin(pos[2]*np.pi/180-np.pi/2),pos[1]-senDist/2*cos(pos[2]*np.pi/180-np.pi/2)]
    rightSen=list(map(lambda x: int((x+1)*512/2),rightSen))
    leftSen=list(map(lambda x: int((x+1)*512/2),leftSen))
    sensorValue=[ph[rightSen[1],rightSen[0],0],ph[leftSen[1],leftSen[0],0]]
    return sensorValue
####################################################################################################################
def devideStates(inp):
    for i in range(len(inp)):
        if abs(inp[i]-255)<5:
            inp[i]=0
        elif abs(inp[i]-250)<5:
            inp[i]= 1
        elif abs(inp[i]-240)<250-240:
            inp[i]= 2
        elif abs(inp[i]-225)<240-225:
            inp[i]= 3
        elif abs(inp[i]-204)<225-204:
            inp[i]= 4
        elif abs(inp[i]-176)<204-176:
            inp[i]= 5
        elif abs(inp[i]-142)<176-142:
            inp[i]= 6
        elif abs(inp[i]-103)<142-103:
            inp[i]= 7
        elif abs(inp[i]-69)<103-69:
            inp[i]= 8
        else: 
            inp[i]= 9
    return inp
####################################################################################################################
class Agent :
    def __init__(self):
        random.seed(21)
        self.timestp=0.1
        self.alpha = 0.1
        self.gamma = 0.01
        self.epsilon = 0.1
        # self.actions={0: [0,0],\
        #               1: [2,0],\
        #               2: [0,2],\
        #               3: [2,2]}

        # self.actions={0: [0,0],\
        #               1: [0,0],\
        #               2: [0,0],\
        #               3: [0,0]}

        self.actions={0: [0,0],\
                      1: [3,1],\
                      2: [1,3],\
                      3: [2,2]}

        self.actionName={0:'O STOP O',1:'O RIGHT >>',2:'<< LEFT O',3:'<< FORWARD >>'}
        self.state_space=10**6
        self.action_space=len(self.actions)
        self.q_table=np.zeros([self.state_space, self.action_space])
        self.penalties=0
    
    def getState(self):
        self.states=[]
        self.states.append(devideStates(stnd_cue()))
        delay(self.timestp)
        self.states.append(devideStates(stnd_cue()))
        delay(self.timestp)
        self.states.append(devideStates(stnd_cue()))
        self.states=np.array(self.states,dtype='int32')
        self.stateIndx=self.states.reshape([1,self.states.size])[0]
        self.state=grandList.index(tuple(self.stateIndx))

    def getNextState(self):
        self.states=[]
        test=[]

        self.states.append(devideStates(stnd_cue()))
        test.append(stnd_cue())
        delay(self.timestp)

        self.states.append(devideStates(stnd_cue()))
        test.append(stnd_cue())
        delay(self.timestp)

        self.states.append(devideStates(stnd_cue()))
        test.append(stnd_cue())
        delay(self.timestp)

        self.states=np.array(self.states,dtype='int32')
        self.stateIndx=self.states.reshape([1,self.states.size])[0]
        self.nextState=grandList.index(tuple(self.stateIndx))
        # print("nextState ",self.nextState," test ",test)

    def decide(self):
        if random.uniform(0, 1) < self.epsilon:
            self.selectedAction = random.randint(0,len(self.actions)-1)
            self.decisionBase=c("explored","red")
        else:
            self.selectedAction = np.argmax(self.q_table[self.state]) 
            self.decisionBase=c("Exploited",'green')

    def act(self):
        L8.set(1)
        motors[0].setVelocity(self.actions[self.selectedAction][0])
        motors[1].setVelocity(self.actions[self.selectedAction][1])
        if self.selectedAction==1: L1.set(1)
        if self.selectedAction==2: L1.set(7)
        if self.selectedAction==3: L1.set(0)
    
    def calculateReward(self):
        # self.reward=np.binary_repr(self.nextState).count('1')-6
        # if self.selectedAction==0:
        #     self.reward=-100
        rewardScaler=100
        rotRewardWeight=2
        posRewardWeight=1
        situation=cord()
        self.rotReward=-1*rotRewardWeight*min(round(abs(situation[2])/180*rewardScaler,2),rewardScaler)
        self.posReward=-1*posRewardWeight*min(round(abs(situation[0])/0.05*rewardScaler,2),rewardScaler)
        self.reward=round(self.rotReward+self.posReward,2)
        if self.reward>=-10: self.reward=10
        if self.selectedAction==0:
            self.reward=-100

    def updateTable(self):
        old_value = self.q_table[self.state, self.selectedAction]
        next_max = np.max(self.q_table[self.nextState])
        # new_value = (1 - self.alpha) * old_value + self.alpha * (self.reward + self.gamma * next_max)
        new_value = old_value + self.alpha * (self.reward + self.gamma * next_max)

        new_value=round(new_value,2)
        self.q_table[self.state, self.selectedAction] = new_value

        # if str(self.state).count('4')>=6 and self.reward <0 and self.selectedAction==3:
        #     print("-----\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        #     print("\n[+] "," state: ",self.state," nextState: ",self.nextState)
        #     print(" action: ",c(self.actionName[self.selectedAction],'blue')," reward ",c(self.reward,'yellow')," posReward ",self.posReward," rotReward ",self.rotReward)
        #     print(" new_value ",new_value," old_value ",old_value," changed ",c(round(new_value-old_value,2),'magenta'))
        #     print(" penalties ",self.penalties," q_table row ",self.q_table[self.state],"\n-------------------------------------------------------------\n")
        #     print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n-----")
        #     stop()
        #     exit()

        if str(self.state).count('4')>=3:
            print("[+] ",self.decisionBase," state: ",self.state," nextState: ",self.nextState)
            print(" action: ",c(self.actionName[self.selectedAction],'blue')," reward ",c(self.reward,'yellow')," posReward ",self.posReward," rotReward ",self.rotReward)
            print(" new_value ",new_value," old_value ",old_value," changed ",c(round(new_value-old_value,2),'magenta'))
            print(" penalties ",self.penalties," q_table row ",self.q_table[self.state],"\n-------------------------------------------------------------")
        self.state=self.nextState
        if self.reward == -100:
            self.penalties += 1

        # state = next_state
        # epochs += 1

    def reset(self):
        if self.state==999999 and self.nextState==999999:
            stop()
            delay(1)
            robot.setCustomData('reset')
            print(c("||||||||||||||||||||||||||||||rested||||||||||||||||||||||||||||||","cyan"))   
            while robot.step(timestep) != -1:
                if robot.getCustomData()!='reset': break
            delay(1)
            self.getState()
    def saveData(self):
        # logname
        with open("_____"+ctime(time()).replace(':','_')+'.npy', 'wb') as f:
            np.save(f,self.q_table)
        print("---------------------------------- saved -----------------------------------")

    
    
    
    
    
    
        
  