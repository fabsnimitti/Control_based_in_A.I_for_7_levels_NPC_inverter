import time
import random
import math
import numpy as np
import time
from ai import DQNAgent# import the class DQNAgent from the archive ai.py, this class was created to be used like a "brain" and to learn with the trial and error

class Inverter():#created the class Inverter used to get the inverter levels
    def __init__(self, level0, level1, level2, level3):
        self.level0= level0
        self.level1= level1
        self.level2= level2
        self.level3= level3
    
    def getLevel(self,level):
        if(level==6):
            return -self.level3
        elif(level==5):
            return -self.level2
        elif(level==4):
            return -self.level1  
        elif(level==3):
            return self.level0
        elif(level==2):
            return self.level1
        elif(level==1):
            return self.level2
        elif(level==0):
            return self.level3
        
class Reward(): # Created the class reward, used to get the reward of each action of the agent
    def getReward(self, state, next_state):
        self.state=state
        self.next_state=next_state
        if(self.state==[0,0] and self.next_state==[0,100]):
            return 1
        elif(self.state==[0,100] and self.next_state==[100,200]):
            return 1
        elif(self.state==[100,200] and self.next_state==[200,300]):
            return 1
        elif(self.state==[200,300] and self.next_state==[300,300]):
            return 1
        elif(self.state==[300,300] and self.next_state==[300,200]):
            return 1
        elif(self.state==[300,200] and self.next_state==[200,100]):
            return 1
        elif(self.state==[200,100] and self.next_state==[100,0]):
            return 1
        elif(self.state==[100,0] and self.next_state==[0,-100]):
            return 1
        elif(self.state==[0,-100] and self.next_state==[-100,-200]):
            return 1
        elif(self.state==[-100,-200] and self.next_state==[-200,-300]):
            return 1
        elif(self.state==[-200,-300] and self.next_state==[-300,-300]):
            return 1
        elif(self.state==[-300,-300] and self.next_state==[-300,-200]):
            return 1
        elif(self.state==[-300,-200] and self.next_state==[-200,-100]):
            return 1
        elif(self.state==[-200,-100] and self.next_state==[-100,0]):
            return 1
        elif(self.state==[-100,0] and self.next_state==[0,100]):
            return 1
        else:
            return -1
if(__name__=="__main__"):    
    inv=Inverter(0,100,200,300) # created the object inv  belong from inverter class  
    rew=Reward()# created the object rew  belong from reward class 
    brain=DQNAgent(2,7)#created the objet brain that belong from the DQNAgent class , this object is used to creat the neural network model and train with the trial and error
    episode=10000# number of episodes
    batch_size=100# batch size(number of sample used to train in each iteration)

    counter=0
    step_by_period=14# step by period, in 50 Hz, 7 levels inverter we have 14 step levels in each period
    last_saved_score=0
    while(counter!=episode):
    
        state=[inv.getLevel(3),inv.getLevel(3)]# "turn on" the inverter
        state = (np.reshape(state, [1, 2]))
        next_state=state #just to initialize this variable
        step_counter=0
        score=0
        
        while(step_counter<=step_by_period):#while for one period
        
            action=brain.act(state)#ask an action to the agent
            next_state=[state[0][1],inv.getLevel(action)] #the state is [last level, actual level]
            next_state=(np.reshape(next_state, [1, 2]))
            reward= rew.getReward([state[0][0], state[0][1]], [next_state[0][0],next_state[0][1]])# get the reward
            score=score+reward # this score is used to know when save the weights
            brain.memorize(state,action,reward,next_state) #emorize each action and reward            
            if(batch_size<len(brain.memory)): #if the size of memory is bigger than batch size, then realize the "replay of experience"
                brain.replay(batch_size)
            if((score>last_saved_score) and (score>10)):# if the score is bigger than last saved score and score is bigger than 10, then save the weights
                brain.save("pesos_score "+str(int(score))+"_" +"ep"+str(counter)+".h5")
                last_saved_score=score
            print("ep: "+str(counter)+" score: "+ str(score)+"  state: "+ str(state) +"   next_sate: " +str(next_state))
            state=next_state
            step_counter=step_counter+1

        counter=counter+1
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    