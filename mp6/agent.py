import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment) # s_prime = none at the start

        # TODO: write your function here
        #Each state in the MDP is a tuple (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)     
        
        if self.train and self.a != None and self.s != None:
            if dead: reward = -1
            elif points > self.points: reward = 1
            else: reward = -0.1
            
            self.N[self.s][self.a]  = self.N[self.s][self.a] + 1
            
            N_sa = self.N[self.s][self.a]
            
            alpha = self.C / (self.C + N_sa)
            
            qmax = max(self.Q[s_prime])
            alpha = self.C / (self.C + N_sa) # learning rate
            oldQ = self.Q[self.s][self.a]
            newQ = oldQ + alpha * (reward + self.gamma * qmax - oldQ)
            
            self.Q[self.s][self.a] = newQ
            
        self.s = s_prime
        self.points = points
            
        if(dead):
            self.reset()
            return 0       
                
        f = np.zeros(4)
        for action in self.actions:
            if self.N[s_prime][action] < self.Ne:
                f[action]= 1
            else:
                f[action] = self.Q[s_prime][action]
        #print(f)
        self.a = 3 - np.argmax(np.flip(f))

        
        return self.a
                        
    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        #snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        
        #food
        if food_x == snake_head_x: food_dir_x = 0
        elif food_x < snake_head_x: food_dir_x = 1
        else: food_dir_x = 2
        
        if food_y == snake_head_y: food_dir_y = 0
        elif food_y < snake_head_y: food_dir_y = 1
        else: food_dir_y = 2
        
        
        #adjoining walls
        if snake_head_x - 1 != 0 and snake_head_x + 1 != utils.DISPLAY_WIDTH -1: adjoining_wall_x = 0 # no wall to the left or right
        elif snake_head_x - 1 == 0: adjoining_wall_x = 1 # wall on left
        else: adjoining_wall_x = 2 # wall must be on right
        
        if snake_head_y - 1 != 0 and snake_head_y + 1 != utils.DISPLAY_HEIGHT -1: adjoining_wall_y = 0 # no wall to the bottom or top
        elif snake_head_y - 1 == 0: adjoining_wall_y = 1 # wall on top
        else: adjoining_wall_y = 2 # wall must be on bottom
           
        #adjoining body
        #[adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for xcord, ycord in snake_body:
            if snake_head_y - 1 == ycord and snake_head_x == xcord: adjoining_body_top = 1 # TOP
            
            if snake_head_y + 1 == ycord and snake_head_x == xcord: adjoining_body_bottom = 1 # BOTTOM
            
            if snake_head_x - 1 == xcord and snake_head_y == ycord: adjoining_body_left = 1 # LEFT
            
            if snake_head_x + 1 == xcord and snake_head_y == ycord: adjoining_body_right = 1 # RIGHT
        
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)       

    
    
    