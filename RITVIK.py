import cv2
import numpy as np
from math import dist
import matplotlib.pyplot as plt

       

class Node:
    def __init__(self, pos, theta, parent, cost, cost2go = 0):
        self.pos = pos
        self.theta = theta
        self.parent = parent
        self.cost = cost
        self.cost2go = cost2go

    def __lt__(self, other):
        return self.cost + self.cost2go < other.cost + other.cost2go
    
    @property
    def key(self):
        return int(self.pos[0]), int(self.pos[1]), int(self.theta//30)

def check_goal(node, goal):
    # print(node.pos, goal.pos)
    
    dt = dist(node.pos, goal.pos)     

    return dt < 1.5        

       
def A_Star(graph, start_node, goal_node):
    ''' Search for a path from start to given goal position
    '''
    # print(start_node.pos, goal_node.pos)
    explored = {start_node.key : start_node}
    cost_dict = {start_node.key: dist(start_node.pos, goal_node.pos)}
    # Run A_Star
    closed = {start_node.key}
    while len(cost_dict):
        key = min(cost_dict, key = cost_dict.get)
        cost_dict.pop(key)
        curr_node = explored[key]
        closed.add(key)
        # if the new_node is goal backtrack and return path
        if check_goal(curr_node, goal_node):
            return backtrack(graph, curr_node), explored
        # get all children of the current node
        for move in graph.moves:
            child_node, new_cost = graph.do_action(curr_node, move, step_size)
            if child_node.key in explored:
                child_node = explored[child_node.key]
            # if the new position is not free, skip the below steps
            if not graph.is_free(child_node.pos) or child_node.key in closed: 
                continue
            # update cost and modify cost_dict
            if new_cost < cost_dict.get(child_node.key, np.inf):
                child_node.parent = curr_node
                child_node.cost = new_cost 
                cost_dict[child_node.key] = new_cost + dist(child_node.pos, goal_node.pos)
                explored[child_node.key] = child_node
    # return None if no path is found.
    return None, None

class Graph:
    
    moves = [60, 30, 0, -30, 60] 

    def __init__(self):
        # size of the map
        self.size = (400, 250)
        self.create_map()
    
    def create_map(self):

        x, y  = np.indices((400, 250))
            
        s1 = y-(0.316*x)-180.26-((0.577*clearance)**2 + (clearance)**2)**(1/2)
        s2 = y+(1.23*x)-218.19 +((0.577*clearance)**2 + (clearance)**2)**(1/2)
        s3 = y+(3.2*x)-457 -((0.577*clearance)**2 + (clearance)**2)**(1/2)
        s4 = y-(0.857*x)-102.135 + ((0.577*clearance)**2 + (clearance)**2)**(1/2)
        s5 = y+(0.1136*x)-189.09 + ((0.577*clearance)**2 + (clearance)**2)**(1/2)

        C = ((y -185)**2) + ((x-300)**2) - (40+5+clearance+rob_radius)**2  
        
        h1 = (y-5) - 0.577*(x+5) - 24.97-((0.577*clearance)**2 + (clearance)**2)**(1/2)
        h2 = (y-5) + 0.577*(x-5) - 255.82 - ((0.577*clearance)**2 + (clearance)**2)**(1/2)
        h3 = (x-6.5 - clearance - rob_radius) - 235 
        h6 = (x+6.5 + clearance + rob_radius) - 165 
        h5 = (y+5) + 0.577*(x+5) - 175 + ((0.577*clearance)**2 + (clearance)**2)**(1/2)
        h4 = (y+5) - 0.577*(x-5) + 55.82 + ((0.577*clearance)**2 + (clearance)**2)**(1/2)

        self.map = ((h1 < 0) & (h2<0) & (h3<0) & (h4>0) & (h5>0) & (h6>0)) | (C<=0)  | ((s1<0) & (s5>0) & (s4>0)) | ((s2>0) & (s5<0) & (s3<0))
        self.map[:6, :] = 1; self.map[:, -6:] = 1
        self.map[:, :6] = 1; self.map[-6:, :] = 1


            
    def is_free(self, pos):
        # inside the map size
        if pos[0] < 0 or pos[1] < 0: 
            return False 
        if pos[0] >= self.size[0] or pos[1] >= self.size[1]:  
            return False 
        # and node is not closed or obstacle
        return self.map[pos] == 0 
    
    
    def do_action(self, node, action, step_size):
        # add the action step to node position to get new position
        x, y = node.pos
        theta = (node.theta + action)%360
        x = x + (step_size*np.cos(np.radians(theta)))
        y = y + (step_size*np.sin(np.radians(theta)))
        x = round(x)
        y = round(y)
        new_cost = 1 + node.cost
        new_node = Node((x, y), theta, node, new_cost)
        return new_node, new_cost
 
    
    def get_mapimage(self):
        img = np.full(self.size, 255, np.uint8)
        obs = np.where(self.map == 1)
        img[obs] = 50
        img = cv2.flip(cv2.transpose(img), 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

def backtrack(graph, node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    # print(path)
    return path
    

def visualize (path, explored):
    ''' Visualise the exploration and the recently found path
    '''
    img = graph.get_mapimage()
    track = path 
    h, w, _ = img.shape
    out = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (w, h))
    
    i = 0
    for key, node in explored.items():
        pos = ( node.pos[0], 249 - node.pos[1])
        parent = node.parent
        if parent is None: parent = node
        pos2 = (parent.pos[0],249 - parent.pos[1])
        cv2.arrowedLine(img, pos, pos2, [0, 80, 0], 2)
        if i%20 == 0:
            out.write(img)
            #out.write(track)
            cv2.imshow('hi', img)
            cv2.waitKey(1)
        i += 0
    

    # for pos in path:
    #     pos = (249 - pos[1], pos[0])
    #     img[pos] = [0, 0, 255]
    # for i in range(50): 
    #     out.write(img)
    out.release()
    cv2.imshow('hi', img)
    cv2.waitKey(0)
           

# when running as main 
if __name__ == '__main__':
    #give start and goal states and orientation and robot radius & clearance 

    
    clearance = int ( input ("Enter the clearance to be given --> ") )
    rob_radius = int ( input ("Enter the radius of the robot --> ") ) 
    step_size = int ( input ("Enter the step_size of the robot --> ") )
    graph = Graph()

    start = input("Enter start coordinates: (seperated by space and hit enter) ")
    start_x, start_y = start.split()
    start = int(start_x), int(start_y)
    start_theta = int (input ("Enter the starting orientation angle-->"))
    if not graph.is_free(start):
        print("In valid start node or in obstacle space")
        exit(-1)
    
    goal = input("Enter goal coordinates: (seperated by space and hit enter) ")
    goal_x, goal_y = goal.split()
    goal= int(goal_x), int(goal_y)
    goal_theta = int( input ("Enter the goal orientation angle -->") ) 

    
    if not graph.is_free(goal):
        print("In valid goal node or in Obstacle space")
        exit(-1)

    start_node = Node(start, start_theta, None, 0)
    goal_node = Node(goal, goal_theta, None, 0)



    path, explored = A_Star(graph, start_node, goal_node)
    
    img = graph.get_mapimage()

    visualize (path, explored)
    # print('\nThe sequence of actions from given start to goal is:')
    # print(solution, '\n')