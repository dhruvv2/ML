# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq

# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze
def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)

def astar(maze, ispart1=False):
    """
    This function returns an optimal path in a list, which contains the start and objective.

    @param maze: Maze instance from maze.py
    @param ispart1:pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    @return: a path in the form of a list of MazeState objects. If there is no path, return None.
    """
    # Your code here
    visited_states = {maze.getStart(): (None, 0)}
    path = []

    heapq.heappush(path, maze.getStart())

    while(path):
        current = heapq.heappop(path)

        if current.is_goal():
            return backtrack(visited_states, current)
        
        

        neighbors = current.get_neighbors(ispart1)
        for neighbor in neighbors:
            if neighbor not in visited_states:
                visited_states[neighbor] = (current, neighbor.dist_from_start)
                heapq.heappush(path, neighbor)
            elif neighbor.dist_from_start < visited_states[neighbor][1]:
                visited_states[neighbor] = (current, neighbor.dist_from_start)
                heapq.heappush(path, neighbor)

    return None

# This is the same as backtrack from MP2
def backtrack(visited_states, current_state):
    path = []
    current = current_state

    while(visited_states[current][0] is not None):
        path.append(current)
        current = visited_states[current][0]

    path.append(current)
    path.reverse()
    return path
        