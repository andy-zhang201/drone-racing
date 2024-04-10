"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np
import matplotlib.pyplot as plt


### PLANNER ###
class RRTStarPlanner:
    def __init__(self, aera_map, start_pos, goal_pos, maxItrs, stepSize, rewireRadius, goalTolerance, collisionTolerance):
        self.aera_map = aera_map
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.maxItrs = maxItrs
        self.stepSize = stepSize
        self.rewireRadius = rewireRadius
        self.goalTolerance = goalTolerance
        self.obstacleList = list()
        self.nodeList = [self.start_pos]
        self.pathFound = False
        self.minDist = float('inf')
        self.nearestNodeIdx = -1
        self.collisionTolerance = collisionTolerance
        self.neighborIdxList = list()

    def Reset(self):
        self.nodeList = [self.start_pos]
        self.pathFound = False
        self.minDist = float('inf')
        self.nearestNodeIdx = -1

    def ResetObstacles(self):
        self.obstacleList = list()

    def AddObstacles(self, obstacle):
        self.obstacleList.append(obstacle)

    def AddChild(self, pos_x, pos_y, parent, cost):
        self.nodeList.append(TreeNode(pos_x, pos_y, parent, cost))

    def GetSteerCoorindate(self, nodeIdx, point):
        if self.nodeList[nodeIdx].pos_x == -0.9889568473925153:
            pass
            # breakpoint()
        
        offset = self.stepSize * FindDirection(self.nodeList[nodeIdx].pos_x, self.nodeList[nodeIdx].pos_y, point[0], point[1])
        return np.array([self.nodeList[nodeIdx].pos_x + offset[0], self.nodeList[nodeIdx].pos_y + offset[1]])

    def MakeSample(self, cur_iter):
        return self.aera_map.MakeSample(cur_iter)

    def CheckGoal(self, currNode):
        return currNode.GetDistanceNode(self.goal_pos) < self.goalTolerance

    def FindNearest(self, rootIdx, point):
        if rootIdx < 0 or rootIdx >= len(self.nodeList):
            return

        dist = self.nodeList[rootIdx].GetDistancePoint(point[0], point[1])
        if dist < self.minDist:
            self.nearestNodeIdx = rootIdx
            self.minDist = dist

        for child in self.nodeList[rootIdx].children:
            self.FindNearest(child, point)


    def CheckCollision(self, currPoint, node):
        # check obstacle
        for ob in self.obstacleList:
            # self.collisionTolerance
            if ob.DetectLineCollision(currPoint[0], currPoint[1], node.pos_x, node.pos_y, self.collisionTolerance):
                return False
        return True

    def FormPath(self):
        # check the last node
        if not self.CheckGoal(self.nodeList[-1]):
            print("not a valid end point!")
            return list()
        parentIdx =self.nodeList[-1].parent
        # add the last element
        tmp = [len(self.nodeList) - 1]
        while parentIdx >= 0:
            tmp.append(parentIdx)
            parentIdx = self.nodeList[parentIdx].parent

        # flip
        res = list()
        n = len(tmp)
        for i in range(n):
            res.append(tmp[n - i - 1])

        return res

    def FindNeighbors(self, currentPoint):
        # find the neighbour with lowest cost
        self.neighborIdxList.clear()
        n = len(self.nodeList)
        for i in range(n):
            dist = self.nodeList[i].GetDistancePoint(currentPoint[0], currentPoint[1])
            if dist <= self.rewireRadius and self.CheckCollision(currentPoint, self.nodeList[i]):
                # store the idx and cost
                self.neighborIdxList.append([i, dist+self.nodeList[i].cost])

    def RewireParent(self, currentPoint):
        if self.nearestNodeIdx == -1:
            print("invalid nearest point!")
            return
        # get the current cost
        currentCost = self.nodeList[self.nearestNodeIdx].cost + self.nodeList[self.nearestNodeIdx].GetDistancePoint(currentPoint[0], currentPoint[1])
        newParent = self.nearestNodeIdx
        # iterate through all neighbors to find a node with lower cost
        for idx in self.neighborIdxList:
            if idx[1] < currentCost:
                newParent = idx[0]
                currentCost = idx[1]
        return newParent, currentCost


    def UpdateOneStep(self, cur_iter, goal):
        # reset nearest value
        self.nearestDist = 1000000
        # do sample
        newPoint = self.MakeSample(cur_iter)
        print(goal)
        if goal:
            newPoint = np.array([goal.pos_x, goal.pos_y])
        
        # find the nearest node in the tree
        self.minDist = float('inf')
        self.nearestNodeIdx = -1
        self.FindNearest(0, newPoint)
        print(f"Nearest Node: ({self.nodeList[self.nearestNodeIdx].pos_x}, {self.nodeList[self.nearestNodeIdx].pos_y})")
        if self.nearestNodeIdx == -1:
            print("can not find nearest point!")
            return
        # do the steering
        next = self.GetSteerCoorindate(self.nearestNodeIdx, newPoint)
        # check obstacle
        
        if not self.CheckCollision(next, self.nodeList[self.nearestNodeIdx]):
            # breakpoint()
            return False

        # get neighbors
        self.FindNeighbors(next)
        # rewire parent
        newParent, newCost = self.RewireParent(next)
        
        # add child
        self.AddChild(next[0], next[1], newParent, newCost)
        self.nodeList[newParent].children.append(len(self.nodeList))
        return True

    def Calculate(self):
        # iterate until whether the maximum itr reached
        for _ in range(self.maxItrs):
            self.UpdateOneStep(_,False)
            if self.CheckGoal(self.nodeList[-1]):
                parentIdx = len(self.nodeList) - 1
                self.pathFound = True
                self.nodeList.append(self.goal_pos)
                self.nodeList[-1].parent = parentIdx
                break
        
        if self.pathFound:
            print("can not find path to goal!")
        else:
            print("path found!")


### UTILS ###


# find the direction of the line given starting and ending points
def FindDirection(starting_x, starting_y, ending_x, ending_y):
    start = np.array([starting_x, starting_y])
    end = np.array([ending_x, ending_y])
    d = end - start

    return d / np.linalg.norm(d)


# use linked list as the node data structure.
class TreeNode:
    def __init__(self, pos_x, pos_y, parent=-1, cost=0):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.children = list()  # initialized the children nodes as an empty list
        self.parent = parent  # invalid idx
        self.cost = cost # cost from root to this node

    def GetDistancePoint(self, other_x, other_y):
        return np.sqrt((self.pos_x - other_x)**2 + (self.pos_y - other_y)**2)

    def GetDistanceNode(self, otherNode):
        return self.GetDistancePoint(otherNode.pos_x, otherNode.pos_y)


# for simplicity, use square map
class SquareMap:
    def __init__(self, bottomleft_x, bottomleft_y, topRight_x, topRight_y):
        self.bottomleft_x = bottomleft_x
        self.bottomleft_y = bottomleft_y
        self.topRight_x = topRight_x
        self.topRight_y = topRight_y

    def MakeSample(self, cur_iter):
        x = np.random.uniform(self.bottomleft_x, self.topRight_x)
        y = np.random.uniform(self.bottomleft_y, self.topRight_y)

        #Every 10 iterations sample at goal.
        if cur_iter % 50 == 0:
            return np.array([self.topRight_x, self.topRight_y])

        return np.array([x, y])


# use circle to define obstacles
class Obstacles:
    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def DetectLineCollision(self, starting_x, starting_y, ending_x, ending_y, tolerance):
        # print(f"Collide Check Starting: ({starting_x}, {starting_y}), Ending: ({ending_x}, {ending_y})")

        if self.DetectPointCollision(starting_x, starting_y, tolerance) or self.DetectPointCollision(ending_x, ending_y, tolerance):
            return True
        
        if starting_x == ending_x and starting_y == ending_y:
            print("Invalid line segment!")
            return False
        
        n0 = FindDirection(starting_x, starting_y, ending_x, ending_y)
        l = np.sqrt((starting_x - starting_y)**2 + (ending_x - ending_y)**2)
        p = np.array([self.center_x - starting_x, self.center_y - starting_y])
        tmp = np.inner(p, n0)
        # if tmp is less than l and greater than 0, means that the nearest point in on the line segement
        if tmp>= 0 and tmp <= l:
            dist = p - tmp * n0
            return np.linalg.norm(dist) <= self.radius + tolerance
        
        # if tmp lies outside of the line segement, check the starting and ending points
        return self.DetectPointCollision(starting_x, starting_y, tolerance) or self.DetectPointCollision(ending_x, ending_y, tolerance)


    def DetectPointCollision(self, pos_x, pos_y, tolerance):
        return np.sqrt((self.center_x - pos_x)**2 + (self.center_y - pos_y)**2) <= self.radius + tolerance


def GenerateMapBorder(squareMap):
    x = [squareMap.bottomleft_x, squareMap.topRight_x, squareMap.topRight_x, squareMap.bottomleft_x, squareMap.bottomleft_x]
    y = [squareMap.bottomleft_y, squareMap.bottomleft_y, squareMap.topRight_y, squareMap.topRight_y, squareMap.bottomleft_y]
    return x, y

def GenerateCircles(obstacle, n=30):
    theta = np.linspace(0, 2*np.pi, n)
    x = list()
    y = list()
    for t in theta:
        x.append(obstacle.center_x + np.cos(t)*obstacle.radius)
        y.append(obstacle.center_y + np.sin(t)*obstacle.radius)
    return x, y

### UAV CODE ###
def visualize(planner):
    fig, ax = plt.subplots()

    # Plot start and goal nodes
    ax.plot(planner.start_pos.pos_x, planner.start_pos.pos_y, 'ro')
    ax.plot(planner.goal_pos.pos_x, planner.goal_pos.pos_y, 'bo')

    # Plot edges of the tree
    for node in planner.nodeList:
        if node.parent is not None:
            parent_node = planner.nodeList[node.parent]
            ax.plot([node.pos_x, parent_node.pos_x], [node.pos_y, parent_node.pos_y], 'k-')

    for obs in planner.obstacleList:
        x, y = GenerateCircles(obs)
        ax.plot(x, y, 'orange')

    # Plot final path
    res = planner.FormPath()
    x = [planner.nodeList[nodeIdx].pos_x for nodeIdx in res]
    y = [planner.nodeList[nodeIdx].pos_y for nodeIdx in res]
    ax.plot(x, y, 'r-')
    plt.axis('equal')
    plt.show()


def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x

def make_map_bounds():

    #x,y
    return np.array([[-3.5,3.5],[-3.5,3.5]])

def add_gates(): #Just as big circles for now
    
    '''
          [ 0.5, -2.5, 0, 0, 0, -1.57, 0],      # gate 1
      [ 2.0, -1.5, 0, 0, 0, 0,     0],      # gate 2
      [ 0.0,  0.2, 0, 0, 0, 1.57,  0],      # gate 3
      [-0.5,  1.5, 0, 0, 0, 0,     0]       # gate 4
      '''

    gate1 = np.array([0.5, -2.5,0.5])
    gate2 = np.array([2.0, -1.5,0.5])
    gate3 = np.array([0.0,  0.2,0.5])
    gate4 = np.array([-0.5,  1.5,0.5])
    
    return [gate1, gate2,gate3]

def add_obstacles(): #Small circles
    return []

def add_start():

    #Init state
    return [-1.0, -3.0]

def add_end():
    return [-0.5, 1.5]

def make_plan(start_x=-1.0,start_y=-3.0,end_x=-0.5,end_y=1.5): #Take in start point and endpoints. Endpoints need to pass through the gate.
    #SHould this function be only called once?
    #It should be called every time you make a plan
    #Then it might be inefficient to reuse Longhao's code, because I will be creating new map each time.
    ## Import Planner
    ## import matplotlib.pyplot as plt
    '''
    What do I need? (numpy arrays)
    - Boundary coordinates (np array)
    - Obstacle coordinates + radii (np array)
    - Gate coordinates (list of np array of obstacles)
    - Start point (np array)
    - End point (np array)
    
    What if I need more?
    - Well then the code will break.
    It should be fixable tho. Let's go ahead with this plan.
    '''
    squareMap = SquareMap(-3.5, -3.5, 3.5, 3.5)

    # starting node
    start = TreeNode(start_x, start_y)
    goal = TreeNode(end_x, end_y)
    maxIters = 10000
    step_size = 0.2
    rewire_radius = 0.7
    goal_tolerance = 1
    collision_tolerance = 0.1

    planner = RRTStarPlanner(squareMap, start, goal, maxIters, step_size, rewire_radius, goal_tolerance, collision_tolerance)

    planner.AddObstacles(Obstacles(-0.5, 0, 0.5))
    # planner.AddObstacles(Obstacles(60, 35, 5))

    for _ in range(10000):
        goal_coords=None

        if _ % 10 == 0:
            goal_coords = goal
        
        # print(f"Iteration: {_}")
        if planner.UpdateOneStep(_, goal_coords):
            parentIdx = planner.nodeList[-1].parent
            x = [planner.nodeList[-1].pos_x, planner.nodeList[parentIdx].pos_x]
            y = [planner.nodeList[-1].pos_y, planner.nodeList[parentIdx].pos_y]
            # ax.plot(x, y, color = 'b', linewidth=0.5)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.pause(0.05)
            if planner.CheckGoal(planner.nodeList[-1]):
                planner.pathFound = True
                parentIdx = len(planner.nodeList) - 1
                planner.nodeList.append(planner.goal_pos)
                planner.nodeList[-1].parent = parentIdx
                break

    print(f"Length of Nodes: {len(planner.nodeList)}")
    res = planner.FormPath()
    x = list()
    y = list()
    for nodeIdx in res:
        x.append(planner.nodeList[nodeIdx].pos_x)
        y.append(planner.nodeList[nodeIdx].pos_y)

    visualize(planner)

    return x, y #return waypoints required to reach there. (limit to 10?)
