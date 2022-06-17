# Reinforcement-Learning
Dongseo University - Prof. 강대기


### Description
'Robocop 3 Genesis - DQN' contains pytorch implementation of Deep Q learning for playing RoboCop 3 by Sega Genesis. To train the model, run <code>python robocop_dqn.py --train</code>. The model will be saved every 100 episodes. Therefore, to test the model, we need to specify the epoch of the model, for example: '--model_epoch 400'.

I included the pretrained models inside 'model/model_dqn' directory. During the training process, I saved the model every 100 episodes for a total of 1300 episodes. Among the 13 models, the model from the 400th eposides was the best. Therefore, I suggest to use this model by running 'python robocop_dqn.py --model_epoch 400'. 
