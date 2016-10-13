import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        location=[]
        for i in range(1,7):
        	for j in range(1,9):
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

		rewardlist=pd.DataFrame(index=state, columns=['forward', 'right','left',None])
		self.Qlist=rewardlist.where(rewardlist.notnull(), 0)

        


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
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
        self.state=pro_inputs
        
        # TODO: Select action according to your policy
        action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.Qlist[state,action]=0.6*reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
