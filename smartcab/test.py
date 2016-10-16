import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np

state_dict={}

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		# TODO: Initialize any additional variables here
		location=[]
		next_point=["forward","left","right",None]
		Fs=[True, False]
		Rs=[True,False]
		Ls=[True,False]
		state=[]
		for i in next_point:
			for j in Fs:
				for k in Rs:
					for l in Ls:
						state.append((i,j,k,l))
		ind=0
		for item in state:
			state_dict[item]=ind
			ind=ind+1

		rewardlist=pd.DataFrame(index=range(len(state)), columns=['forward', 'right','left',None])

		with open("Qlist.txt",'r') as f:
			for i in rewardlist.index:
				dataline=f.readline().split('\t')
				for j in range(len(rewardlist.columns)):
					rewardlist.loc[i,rewardlist.columns[j]]=float(dataline[j])
		self.Qlist=rewardlist




	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required

	def update(self, t):

		# Gather inputs
		self.destination=self.env.agent_states[self]["destination"]
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		deadline = self.env.get_deadline(self)
		inputs = self.env.sense(self)
		location=self.env.agent_states[self]['location']
		heading=self.env.agent_states[self]['heading']
		nexts=self.next_waypoint
		light=inputs['light']
		oncoming=inputs['oncoming']
		left=inputs['left']
		right=inputs['right']
		forward_safe=light=='green'
		right_safe=light=='green' or (light=='red' and not oncoming=='left' and not left=='forward')
		left_safe=light=='green' and 'oncoming'==None
		pro_inputs={'Next_waypoint':self.next_waypoint, 'F':forward_safe, 'R':right_safe, 'L':left_safe}
		state=(self.next_waypoint, forward_safe, right_safe,left_safe)
		self.state=state


		#Choose the best action
		best_Q=self.Qlist.loc[state_dict[self.state]].max()
		action_l=[]
		for item in ["forward","left","right",None]:
			if self.Qlist.loc[state_dict[self.state],item]==best_Q:
				action_l.append(item)

		action = random.choice(action_l)
		# action=random.choice(["forward","right","left",None])

		# Execute action and get reward
		reward = self.env.act(self, action)

		#Get next state		
		# TODO: Learn policy based on state, action, reward
		print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)



def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False


    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print e.success_time


if __name__ == '__main__':
    run()
