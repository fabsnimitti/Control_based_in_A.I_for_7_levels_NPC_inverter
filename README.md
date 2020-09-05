# Control based in A.I for 7 levels NPC nverter
In this repositorie you can found a stable control for 7 levels inverter, there are 3 files called ai.py, test_npc.py and NPC_7.py.
# The ai.py file:
Contain the "brain" of this project, the neural network model and some function to realize the exprience replay and memorize the action.
# The test_npc.py file:
it is a file created to test and plot the inverter output, it load the model and weights and make a test for one period of 50Hz sine wave.
# The NPC_7.py file:
It is our training environment, this file work together with ai.py to create and train our model.
# Click in this imagem to watch the training video.
[![vide](https://img.youtube.com/vi/7UqTWPhouaw/0.jpg)](https://www.youtube.com/watch?v=7UqTWPhouaw)
# The output that this A.I got:
We can see that this project works well, since the output is the same of a complex analog control circuit as you can see below.


![output](https://github.com/fabsnimitti/Control_based_in_A.I_for_7_levels_NPC_inverter/blob/master/Output/output.png)


