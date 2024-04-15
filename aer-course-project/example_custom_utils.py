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
        
        offset = self.stepSize * FindDirection(self.nodeList[nodeIdx].pos_x, self.nodeList[nodeIdx].pos_y, point[0], point[1])
        return np.array([self.nodeList[nodeIdx].pos_x + offset[0], self.nodeList[nodeIdx].pos_y + offset[1]])

    def MakeSample(self):
        return self.aera_map.MakeSample()
    
    def MakeSampleSubArea(self):
        buffer = 0.2

        # Determine the minimum and maximum x values, accounting for the buffer
        x_min = min(self.start_pos.pos_x, self.goal_pos.pos_x) - buffer
        x_max = max(self.start_pos.pos_x, self.goal_pos.pos_x) + buffer

        # Determine the minimum and maximum y values, accounting for the buffer
        y_min = min(self.start_pos.pos_y, self.goal_pos.pos_y) - buffer
        y_max = max(self.start_pos.pos_y, self.goal_pos.pos_y) + buffer

        # Sample uniformly within the ranges
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)

        return np.array([x_sample, y_sample])


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
        # breakpoint()
        # This if statement NEVER triggers. Could goal never be checked for collision?

        # if self.CheckGoal(node) or ((abs(currPoint[0] - self.goal_pos.pos_x) < 0.8) and (abs(currPoint[1] - self.goal_pos.pos_y) < 0.8)):
            # print(f"Checking collision with Goal: ({node.pos_x}, {node.pos_y})")
            # print(f"Other point: ({currPoint[0]}, {currPoint[1]})")
            # print("=====================================")
            # print(f"Goal point: ({self.goal_pos.pos_x}, {self.goal_pos.pos_y})")
            # breakpoint()
    
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


    def UpdateOneStep(self, goal):
        # reset nearest value
        self.nearestDist = 1000000
        # do sample in subarea around goal and start


        newPoint = self.MakeSampleSubArea()

        if goal:
            newPoint = np.array([goal.pos_x, goal.pos_y])
        
        # find the nearest node in the tree
        self.minDist = float('inf')
        self.nearestNodeIdx = -1
        self.FindNearest(0, newPoint)
        # print(f"Nearest Node: ({self.nodeList[self.nearestNodeIdx].pos_x}, {self.nodeList[self.nearestNodeIdx].pos_y})")
        if self.nearestNodeIdx == -1:
            print("can not find nearest point!")
            return
        # do the steering
        next = self.GetSteerCoorindate(self.nearestNodeIdx, newPoint)
        # check obstacle
        
        if not self.CheckCollision(next, self.nodeList[self.nearestNodeIdx]):
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
            self.UpdateOneStep(False)
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

    def MakeSample(self):
        x = np.random.uniform(self.bottomleft_x, self.topRight_x)
        y = np.random.uniform(self.bottomleft_y, self.topRight_y)

        return np.array([x, y])


# use circle to define obstacles
class Obstacles:
    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def DetectLineCollision(self, starting_x, starting_y, ending_x, ending_y, tolerance):
        # # print(f"Collide Check Starting: ({starting_x}, {starting_y}), Ending: ({ending_x}, {ending_y})")

        if self.DetectPointCollision(starting_x, starting_y, tolerance) or self.DetectPointCollision(ending_x, ending_y, tolerance):
            return True
        
        if starting_x == ending_x and starting_y == ending_y:
            print("Invalid line segment!")
            return True
        
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
            ax.plot([node.pos_x, parent_node.pos_x], [node.pos_y, parent_node.pos_y], 'k-',linewidth = 1)

    idx = 0
    for obs in planner.obstacleList:
        if planner.obstacleList[idx] == obs:
            x, y = GenerateCircles(obs)
            if idx > 3:
                ax.plot(x, y, 'green',linewidth='2')
            else:
                ax.plot(x,y,'orange')
            idx = idx+1

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

def make_plan(start_x,start_y,gate_coords):
    squareMap = SquareMap(-3.5, -3.5, 3.5, 3.5)

    # starting node
    start = TreeNode(start_x, start_y)
    maxIters = 10000
    step_size = 0.2
    rewire_radius = 0.5
    goal_tolerance = 0.1
    collision_tolerance = 0.1

    x_total = list()
    y_total = list()

    for gol in gate_coords:
        goal = TreeNode(gol[0], gol[1])
        planner = RRTStarPlanner(squareMap, start, goal, maxIters, step_size, rewire_radius, goal_tolerance, collision_tolerance)

    #planner.AddObstacles(Obstacles(-0.5, 0, 0.5))
        planner.AddObstacles(Obstacles(1.5, -2.5, 0.06))
        planner.AddObstacles(Obstacles(0.5, -1, 0.06))
        planner.AddObstacles(Obstacles(1.5, 0.0, 0.06))
        planner.AddObstacles(Obstacles(-1.0, 0.0, 0.06))

    #gates
        planner.AddObstacles(Obstacles(0.4,-2.3,0.06))
        planner.AddObstacles(Obstacles(0.4,-2.7,0.06))
        planner.AddObstacles(Obstacles(0.5,-2.3,0.06))
        planner.AddObstacles(Obstacles(0.5,-2.7,0.06))
        planner.AddObstacles(Obstacles(0.6,-2.3,0.06))
        planner.AddObstacles(Obstacles(0.6,-2.7,0.06))
        
        planner.AddObstacles(Obstacles(2.2,-1.4,0.06))
        planner.AddObstacles(Obstacles(1.8,-1.4,0.06))
        planner.AddObstacles(Obstacles(2.2,-1.5,0.06))
        planner.AddObstacles(Obstacles(1.8,-1.5,0.06))
        planner.AddObstacles(Obstacles(2.2,-1.6,0.06))
        planner.AddObstacles(Obstacles(1.8,-1.6,0.06))

        planner.AddObstacles(Obstacles(-0.1,0.4,0.06))
        planner.AddObstacles(Obstacles(-0.1,0.0,0.06))
        planner.AddObstacles(Obstacles(0.0,0.4,0.06))
        planner.AddObstacles(Obstacles(0.0,0.0,0.06))
        planner.AddObstacles(Obstacles(0.1,0.4,0.06))
        planner.AddObstacles(Obstacles(0.1,0.0,0.06))

        planner.AddObstacles(Obstacles(-0.3,1.4,0.06))
        planner.AddObstacles(Obstacles(-0.7,1.4,0.06))
        planner.AddObstacles(Obstacles(-0.3,1.5,0.06))
        planner.AddObstacles(Obstacles(-0.7,1.5,0.06))
        planner.AddObstacles(Obstacles(-0.3,1.6,0.06))
        planner.AddObstacles(Obstacles(-0.7,1.6,0.06))


    # planner.AddObstacles(Obstacles(60, 35, 5))

        for _ in range(maxIters):
            goal_coords_sampling=None

            if _ % 10 == 0:
                goal_coords_sampling = goal
        
        # print(f"Iteration: {_}")
            if planner.UpdateOneStep(goal_coords_sampling):
                parentIdx = planner.nodeList[-1].parent
                x = [planner.nodeList[-1].pos_x, planner.nodeList[parentIdx].pos_x]
                y = [planner.nodeList[-1].pos_y, planner.nodeList[parentIdx].pos_y]

                if planner.CheckGoal(planner.nodeList[-1]):
                    planner.pathFound = True
                    parentIdx = len(planner.nodeList) - 1
                    planner.nodeList.append(planner.goal_pos)
                    planner.nodeList[-1].parent = parentIdx
                    break

        # print(f"Length of Nodes: {len(planner.nodeList)}")
        res = planner.FormPath()
        visualize(planner)
   

        #Update Start Location
        old_start = start
        start = TreeNode(gol[0], gol[1])

        #Append waypoints
        x_sub = list()
        y_sub = list()
        for nodeIdx in res:
            x_sub.append(planner.nodeList[nodeIdx].pos_x)
            y_sub.append(planner.nodeList[nodeIdx].pos_y)
        

        #Add extra waypoints in front and behind gate to ensure traj doesn't hit gate
        """
        Need:
        1. Direction of gate (found using gol[-2])
        2. Location of start (start) and end (gol [0], gol[1])
        3. Buffer distance
        
        """

        buffer_distance = 0.25
        direction = gol[-2]
        beginning = np.array([old_start.pos_x, old_start.pos_y])
        ending = np.array([gol[0], gol[1]])
        

        # If goal is facing north/south
        if (direction == 0):
            if (gol[1] > beginning[1]):
                x_front = gol[0] 
                y_front = gol[1] - buffer_distance
                x_back = gol[0] 
                y_back = gol[1] + buffer_distance
            else:
                x_front = gol[0] 
                y_front = gol[1] + buffer_distance
                x_back = gol[0] 
                y_back = gol[1] - buffer_distance

            #delete any waypoints which come in between the back and front waypoints
            idx = 0
            while idx < len(x_sub):
                if (y_sub[idx] > min(y_front,y_back)) and (y_sub[idx] < max(y_front,y_back)):
                    del x_sub[idx]
                    del y_sub[idx]

                else:
                    idx += 1               

        #If goal is facing east/west
        elif((direction == -1.57) or (direction == 1.57)):
            if (gol[0] > beginning[0]):
                x_front = gol[0] - buffer_distance
                y_front = gol[1]
                x_back = gol[0] + buffer_distance
                y_back = gol[1]
            else:
                x_front = gol[0] + buffer_distance
                y_front = gol[1]
                x_back = gol[0] - buffer_distance
                y_back = gol[1]
            
            #delete any waypoints which come in between the back and front waypoints
            idx = 0
            while idx < len(x_sub):
                x = x_sub[idx]
                if (x > min(x_front, x_back)) and (x < max(x_front, x_back)):
                    del x_sub[idx]
                    del y_sub[idx]

                else:
                    idx += 1


        else:
            pass

        # insert waypoints into second to last element of x_sub and y_sub
        x_sub.append(x_front)
        y_sub.append(y_front)

        #Reinsert goal waypoints
        x_sub.append(gol[0])
        y_sub.append(gol[1])

        # Insert back waypoints
        x_sub.append(x_back)
        y_sub.append(y_back)

        #Concatenate lists
        x_total = x_total + x_sub
        y_total = y_total + y_sub
        #Set next start to the farthest additional waypoint
        start = TreeNode(x_back, y_back)

        # # Add flag to signal end of subproblem
        # x_total.append(-1000)
        # y_total.append(-1000)

    return x_total, y_total #return waypoints required to reach there. (limit to 10?)
