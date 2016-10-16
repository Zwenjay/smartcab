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
		for i in range(1,9):
			for j in range(1,7):
				location.append((i,j))
		heading=[(0,1),(0,-1),(-1,0),(1,0)]
		Fs=[True, False]
		Rs=[True,False]
		Ls=[True,False]
		state=[]
		for i in location:
			for j in heading:
				for k in Fs:
					for l in Rs:
						for m in Ls:
							state.append((i,j,k,l,m))
		ind=0
		for item in state:
			state_dict[item]=ind
			ind=ind+1

		rewardlist=pd.DataFrame(index=range(len(state)), columns=['forward', 'right','left',None])

		with open("Qlist_1000times_a_7.txt",'r') as f:
			for i in rewardlist.index:
				dataline=f.readline().split('\t')
				for j in rewardlist.columns:
					rewardlist.loc[state_dict[i],j]=double(dataline[j])
		self.Qlist=rewardlist




	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required

	def update(self, t):
		print self.Qlist.head()
		# Gather inputs
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
		pro_inputs={'Locaation':location,'Heading':heading, 'F':forward_safe, 'R':right_safe, 'L':left_safe}
		state=(location, heading, forward_safe, right_safe,left_safe)
		self.state=state


		#Choose the best action
		best_Q=self.Qlist.loc[state_dict[current_state]].max()
		action_l=[]
		for item in ["forward","left","right",None]:
			if self.Qlist.loc[state_dict[current_state],item]==best_Q:
				action_l.append(item)
		action = random.choice(action_l)

		current_state=self.state

		# Execute action and get reward
		reward = self.env.act(self, action)

		#Get next state
		

		# TODO: Learn policy based on state, action, reward
		self.Qlist.loc[state_dict[current_state],action]=reward+0.7*np.max(self.Qlist.loc[state_dict[next_state]])
		print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)



def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False


    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print e.success_time
    with open('Qlist.txt','w') as f:
		for states in a.Qlist.index:
			for actions in a.Qlist.columns:
				f.write(str(a.Qlist.loc[states,actions]))
				f.write('\t')
			f.write('\r\n')


if __name__ == '__main__':
    run()
