
import os #library used to acess OS names
os.environ["KERAS_BACKEND"] = "theano" # say to OS that i will use theano as keras backend
from keras.models import model_from_json #importation to use json 
import keras 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt #library to plot the result

from NPC_7 import Inverter #import the class Inverter, created in trainer archive
inv=Inverter(0,100,200,300) #create the object inv, used to get inverter levels 

  
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("pesos_score 15_ep177.h5")
loaded_model.compile(loss='mse', optimizer=Adam(lr=0.001))



counter=0.0013 #counter start in 0.013 because for 50Hz and 7 levels inverter, the time of one step level is 0.0013 seconds
states=[] # variable used to save the state
time=[] # variable used to save the time of each state
state=[0,0] # start the first state(turn on the inverter)
states.append(state[1]) #append the first state
time.append(0) #append the time that the inverter start
states.append(state[1]) #again append the start level, but this time means it is the last time for this level
time.append(counter-0.0000000001) #append when is the last time the first level will be activated before the next level be activated

while(counter<=(1/50)):#start while loop to create one period of 50Hz pseud-sine wave by 7 levels inverter
    state = (np.reshape(state, [1, 2]))#state
    action=np.argmax(loaded_model.predict(state)[0])#using the state, neural network predict what is the next level
    state=[state[0][1],inv.getLevel(action)]# update the state to the level predicted
    states.append(state[1])#append the state
    time.append(counter)#append the time of this state
    states.append(state[1])#append again the state
    time.append(counter+0.0013-0.0000000001)#append when is the last time the first level will be activated before the next level be activated
    counter=counter+0.0013#sum the time
  
#plot the graph
plt.plot(time, states)
plt.ylabel('Tensão')
plt.xlabel('Tempo')

plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    