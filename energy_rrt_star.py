# Viranjan Bhattacharyya
# EMC2 Lab March 2022

import numpy as np
import matplotlib.pyplot as plt
import time

""" Elevation aware adaptive d_move """
class VRRTStar:
    class Node:
        def __init__(self, states):
            self.coordinates = states
            self.parent = None
            self.cost = 0

    def __init__(self, start, goal, x_bounds, y_bounds, v_bounds, goal_bias, delta_goal, max_iterations, d_move, r_neighborhood, elevation):
        self.start_ = start
        self.goal_ = goal
        self.x_bounds_ = x_bounds
        self.y_bounds_ = y_bounds
        self.v_bounds_ = v_bounds
        self.goal_bias_ = goal_bias
        self.delta_goal_ = delta_goal
        self.max_iterations_ = max_iterations
        self.d_move_ = d_move
        self.r_neighborhood_ = r_neighborhood
        self.elevation_ = elevation

    def plan(self):
        node_list = [self.Node(self.start_)]        
        for i in range(self.max_iterations_):
            sample = self.samplePoint()
            near_node = self.findNearestNode(sample, node_list)
            new_node = self.addNode(near_node, sample)
            neighbor_list = self.findNeighborhoodNodes(new_node, node_list)            
            new_node = self.chooseParent(new_node, neighbor_list)

            if self.safeIntermediateElevation(new_node):
                self.d_move_ = 3
                node_list.append(new_node)
                self.rewire(new_node, neighbor_list)
            else:
                self.d_move_ = 1
            # print(new_node.coordinates)
            i+=1

        return node_list
    
    def distance(self, p1, p2):
        """ Find the Euclidean distance between two points p1 and p2 """

        d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

        return d    

    def samplePoint(self):
        """
        Samples points in space.
        x_bounds: [x_min, x_max], y_bounds: [y_min, y_max], v_bounds: [v_min, v_max]
        goal_bias: probability of selecting goal point

        """        
        if np.random.rand(1)>self.goal_bias_:
            x_sample = self.x_bounds_[0] + np.random.rand(1)*(self.x_bounds_[1]-self.x_bounds_[0])
            y_sample = self.y_bounds_[0] + np.random.rand(1)*(self.y_bounds_[1]-self.y_bounds_[0])
            v_sample = self.v_bounds_[0] + np.random.rand(1)*(self.v_bounds_[1]-self.v_bounds_[0])
        else:
            x_sample = self.goal_[0]
            y_sample = self.goal_[1]
            v_sample = self.goal_[2]

        sample_node = [x_sample, y_sample, v_sample]
        # print(sample_node)
        return self.Node(sample_node)

    def findNearestNode(self, sample_point, node_list):
        """
        Find distances from sampled point to each point in tree (node_list).
        Return the nearest node.

        """
        distances = []
        p1 = np.array(sample_point.coordinates)

        for i in range(len(node_list)):
            p2 = np.array(node_list[i].coordinates)
            d = self.distance(p1, p2)
            distances.append(d)
        
        distances = np.array(distances)
        min_index = np.argmin(distances)        

        return node_list[min_index]

    def addNode(self, nearest_node, sample_node):
        """
        Point in direction of line joining sampled point and nearest node, at a distance equal to d_move.
        Find nearby discrete point.
        Parent of newly added node is the nearest node to sampled point.

        """
        theta = np.arctan2((sample_node.coordinates[1]-nearest_node.coordinates[1]),(sample_node.coordinates[0]-nearest_node.coordinates[0]))
        new_point = [nearest_node.coordinates[0] + self.d_move_*np.cos(theta), nearest_node.coordinates[1] + self.d_move_*np.sin(theta), sample_node.coordinates[2]]
                
        new_node = self.Node(new_point)
        new_node.parent = nearest_node
        # print(new_node.coordinates)
        return new_node

    def findNeighborhoodNodes(self, node, node_list):
        """ 
        Function to find neighborhood nodes of a given node.
        Searches within r_neighborhood radius.
        Returns a list of neighbors.

        """
        neighbor_list = []
        p1 = np.array(node.coordinates)

        for i in range(len(node_list)):
            p2 = np.array(node_list[i].coordinates)            
            d = self.distance(p1, p2)

            if d <= self.r_neighborhood_:
                neighbor_list.append(node_list[i])

        return neighbor_list
    
    def chooseParent(self, node, neighbor_list):
        """ 
            For a given node and its neighbors, find the node in neighbor_list
            connecting to which results in minimum cost for the given node.
            Set that node in neighbor_list as the parent of the given node.

        """
        
        costs = []
        min_cost = np.inf
        parent = None

        for i in range(len(neighbor_list)):
            c = neighbor_list[i].cost + self.cost(neighbor_list[i].coordinates, node.coordinates)
            costs.append(c)
                
        costs = np.array(costs)
        min_index = np.argmin(costs)                
        parent = neighbor_list[min_index]
        min_cost = costs[min_index]        
        
        node.parent = parent 
        node.cost = min_cost   

        return node

    def cost(self, p1, p2):
        """ 
        Find cost of moving from point p1 to p2.                
            
        """
        # theta = np.arctan2((self.elevation_[int(p2[0]),int(p2[1])] - self.elevation_[int(p1[0]),int(p1[1])]), (self.distance(p1, p2)+0.00001))
        P1 = [p1[0],p1[1],self.elevation_[int(p1[0]),int(p1[1])]]
        P2 = [p2[0],p2[1],self.elevation_[int(p2[0]),int(p2[1])]]

        # J = (abs(p1[2]**2 - p2[2]**2) # KE
        # + 9.81*abs(self.elevation_[int(p1[0]),int(p1[1])] - self.elevation_[int(p2[0]),int(p2[1])]) # PE
        # + 9.81*0.015*self.distance(p1, p2) # Rolling
        # + 0.0004*(p1[2]**2)*self.distance(P1,P2) + max(0, p1[2]*(p1[2] - p2[2])) # Drag + Braking losses ZOH
        # ) 
         
        # J = (abs(p1[2]**2 - p2[2]**2) # KE
        # + 9.81*abs(self.elevation_[int(p1[0]),int(p1[1])] - self.elevation_[int(p2[0]),int(p2[1])]) # PE
        # + 9.81*0.015*self.distance(p1, p2) # Rolling
        # + 0.0004*((((p1[2] + p2[2])/2)**2)*self.distance(P1,P2)) + max(0, ((p1[2] + p2[2])/2)*(p1[2] - p2[2])) # Drag + Braking losses average velocity
        # ) 
         
        J = (0.5*abs(p1[2]**2 - p2[2]**2) # KE
        + 9.81*abs(self.elevation_[int(p1[0]),int(p1[1])] - self.elevation_[int(p2[0]),int(p2[1])]) # PE
        + 9.81*0.015*self.distance(p1, p2) # Rolling
        + 0.0004*((((p2[2] - p1[2])**2)*self.distance(P1,P2))/3 + p1[2]*p2[2]*self.distance(P1,P2)) # Drag loss :: FOH velocity
        + max(0, ((p1[2] + p2[2]))*(p1[2] - p2[2])/2) # Braking loss
        )      
        
        return J

    def rewire(self, node, neighbor_list):
        """
        Check if choosing given node as parent results in 
        lower cost of the nodes in neighbor_list

        """
        for n in neighbor_list:
            if (node.cost + self.cost(node.coordinates, n.coordinates)) < n.cost:
                n.parent = node
                n.cost = node.cost + self.cost(node.coordinates, n.coordinates)

    def drawBranches(self, node_list):
        for node in node_list:
            if node.parent:
                plt.plot([node.coordinates[0], node.parent.coordinates[0]], [node.coordinates[1], node.parent.coordinates[1]], "-m")  

    def goalNode(self, node_list):
        """ Find the lowest cost node to the goal """

        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(node_list)):
            node = node_list[i]
            # Has to be in close proximity to the goal
            if self.distance(node.coordinates, self.goal_) <= self.delta_goal_:                
                # The final path length                
                if (node.cost + self.cost(node.coordinates, self.goal_)) < min_cost:
                    # Found better goal node!
                    min_cost = node.cost + self.cost(node.coordinates, self.goal_) 
                    best_goal_node_idx = i                

        goal_node = node_list[best_goal_node_idx]
        goal_node.cost = min_cost
        return goal_node

    def pathfromGoal(self, goal_node, node_list):
        """ Backtrack path from goal to start  """
        node = goal_node
        path = [node]
        while node != node_list[0]:
            node = node.parent
            path.append(node) 

        return path 
    
    def drawPath(self, goal_node, node_list):
        path = self.pathfromGoal(goal_node, node_list)
        x = []
        y = []
        for n in range(len(path)):
            x.append(path[n].coordinates[0])
            y.append(path[n].coordinates[1])

        plt.plot(x,y,'b') 

    def safeIntermediateElevation(self, new_node):
        """ Function to check if points along d_move have acceptable approach angle """
        px2 = int(new_node.coordinates[0])
        py2 = int(new_node.coordinates[1])       

        parent = new_node.parent
        px1 = int(parent.coordinates[0])
        py1 = int(parent.coordinates[1])

        if px2>px1:
            x2 = px2
            y2 = py2
            x1 = px1
            y1 = py1
        else:
            x2 = px1
            y2 = py1
            x1 = px2
            y1 = py2
        # print(x1,y1,x2,y2)
        slope = [0]
        while x2>x1:
            z2 = self.elevation_[x2,y2]
            z1 = self.elevation_[x1,y1]
            d = 50*np.sqrt((x2-x1)**2 + (y2-y1)**2)
            slope.append((z2-z1)/d)
            x2-=1
            y2-=1
        
        if max(slope)<0.4:
            return True
        else:
            return False

if __name__ == "__main__":
    #########################################################################################
    #                                        Simulation                                     #
    #########################################################################################
    startTime = time.time()

    # np.random.seed(5)
    # start = [0,0,0]
    # goal = [180,180,0]
    start = [180,150,0]
    goal = [600,800,10]
    # x_bounds = [0,200]
    # y_bounds = [0,200]
    x_bounds = [0,1081]
    y_bounds = [0,1081]
    v_bounds = [0,10]
    goal_bias = 0.1
    delta_goal = 10
    max_iterations = 1000
    d_move = 5
    r_neighborhood = 50
    z = np.loadtxt("C:/Users/viran/OneDrive/Documents/VIPR/Python/RRTs-main/map1.txt")
    rrtStar = VRRTStar(start, goal, x_bounds, y_bounds, v_bounds, goal_bias, delta_goal, max_iterations, d_move, r_neighborhood, z)
    tree = rrtStar.plan()

    executionTime = time.time() - startTime
    print('Execution time in seconds: ' + str(executionTime))

    rrtStar.drawBranches(tree)

    goalnode = rrtStar.goalNode(tree)
    rrtStar.drawPath(goalnode, tree)
    plt.plot(start[0],start[1],'go')

    # X = np.arange(0, 200, 1)
    # Y = np.arange(0, 200, 1)
    X = np.arange(0, 1081, 1)
    Y = np.arange(0, 1081, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.contourf(X,Y,z)
    plt.colorbar(fig)

    plt.show()

    sol = rrtStar.pathfromGoal(goalnode, tree)
    l = []
    for i in range(len(sol)):
        l.append(sol[i].coordinates)

    l = np.array(l)
    # print(l)
    np.savetxt("C:/Users/viran/OneDrive/Documents/VIPR/Python/RRTs-main/path1.txt", l)

    # print(goalnode.cost)
    # print(tree[-1].coordinates)
    # print(z[0,0])

    # print(len(tree))
    # 990 nodes in 1000 iterations