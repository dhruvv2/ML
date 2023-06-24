# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP3
"""

import math
import numpy as np
from alien import Alien
from typing import List, Tuple

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """

    granularity_calc = ((granularity)/np.sqrt(2)) + alien.get_width()
    for wall in walls:
        startxy = (wall[0], wall[1])
        endxy = (wall[2], wall[3])
        wall_segment = (startxy, endxy)
        if alien.is_circle():
            wall_dist = point_segment_distance(alien.get_centroid(), wall_segment)
        else:
            wall_dist = segment_distance(alien.get_head_and_tail(), wall_segment)
        if np.isclose(wall_dist, granularity_calc) or wall_dist <= granularity_calc:
            return True

    return False

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """

    for goal in goals:
        goal_segment = (goal[0], goal[1])
        radius = goal[2]
        goal_edge = radius + alien.get_width()
        if alien.is_circle():
            goal_dist = np.sqrt((goal[0] - alien.get_centroid()[0])**2 + (goal[1] - alien.get_centroid()[1])**2)
        else:
            goal_dist = point_segment_distance(goal_segment, alien.get_head_and_tail())
        if np.isclose(goal_dist, goal_edge) or goal_dist <= goal_edge:
            return True


    return False

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    granularity_calc = ((granularity)/np.sqrt(2)) + alien.get_width()

    window_borders = np.array([((0,0),(0,window[1])), ((0,window[1]),(window)), ((window),(window[0], 0)), ((window[0],0),(0,0))])
    for i in range(len(window_borders)):
        if alien.is_circle():
            window_dist = point_segment_distance(alien.get_centroid(), window_borders[i])
            if np.isclose(window_dist, granularity_calc) or window_dist <= granularity_calc:
                return False
        else:
            window_dist = segment_distance(alien.get_head_and_tail(), window_borders[i])
            if np.isclose(window_dist, granularity_calc) or window_dist <= granularity_calc:
                return False

    return True

def point_segment_distance(point, segment):
    """Compute the distance from the point to the line segment.
    Hint: Lecture note "geometry cheat sheet"

        Args:
            point: A tuple (x, y) of the coordinates of the point.
            segment: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """


    # p = np.asarray(point)
    # l1 = np.asarray(segment[0])
    # l2 = np.asarray(segment[1])
    # d = np.abs(np.linalg.norm(np.cross(l2 - l1, l1 - p))/np.linalg.norm(l2 - l1))
    # cross = (np.cross(l2 - l1, l1 - p))
    # # if cross < 0:
    # #     print('hello')
    
    # if cross > 0:
    #     x1_dist = np.sqrt((segment[0][0] - point[0])**2 + (segment[0][1] - point[1])**2)
    #     x2_dist = np.sqrt((segment[1][0] - point[0])**2 + (segment[1][1] - point[1])**2)
    #     d = min(x1_dist, x2_dist)
    #     print('hello')

    x = point[0]
    y = point[1]

    x1 = segment[0][0]
    y1 = segment[0][1]

    x2 = segment[1][0]
    y2 = segment[1][1]

    euclid = ((x2 - x1)**2 + (y2 - y1)**2)

    if euclid == 0:
        part1 = -1
    else:
        part1 = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / euclid
    
    if part1 > 1: 
        part1 = 1 
    elif part1 < 0:
        part1 = 0

    d = np.sqrt((x1 + part1*(x2 - x1) - x)**2 + (y1 + part1*(y2 - y1) - y)**2)
    
    return d

def do_segments_intersect(segment1, segment2):
    """Determine whether segment1 intersects segment2.  
    We recommend implementing the above first, and drawing down and considering some examples.
    Lecture note "geometry cheat sheet" may also be handy.

        Args:
            segment1: A tuple of coordinates indicating the endpoints of segment1.
            segment2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    #above function referenced from geekforgeeks
    #segments p1p2 and p3p4
    p1 = segment1[0]
    p2 = segment1[1]

    p3 = segment2[0]
    p4 = segment2[1]

    v13 = [p1[0] - p3[0], p1[1] - p3[1]]
    v23 = [p2[0] - p3[0], p2[1] - p3[1]]
    v43 = [p4[0] - p3[0], p4[1] - p3[1]]

    v21 = [p2[0] - p1[0], p2[1] - p1[1]]
    v31 = [p3[0] - p1[0], p3[1] - p1[1]]
    v41 = [p4[0] - p1[0], p4[1] - p1[1]]

    or1 = np.cross(v13, v43)
    or2 = np.cross(v23, v43)
    or3 = np.cross(v31, v21)
    or4 = np.cross(v41, v21)

    if ((or1 < 0 and or2 > 0) or (or1 > 0 and or2 < 0)) and ((or3 > 0 and or4 < 0) or (or3 < 0 and or4 > 0)):
        return True

    left_x_seg1 = min(p1[0], p2[0])
    right_x_seg1 = max(p1[0], p2[0])
    left_y_seg1 = min(p1[1], p2[1])
    right_y_seg1 = max(p1[1], p2[1])

    left_x_seg2 = min(p3[0], p4[0])
    right_x_seg2 = max(p3[0], p4[0])
    left_y_seg2 = min(p3[1], p4[1])
    right_y_seg2 = max(p3[1], p4[1])

    if (p1[0] <= right_x_seg2 and p1[0] >= left_x_seg2) and (p1[1] <= right_y_seg2 and p1[1] >= left_y_seg2) and or1 == 0:
        return True

    if (p2[0] <= right_x_seg2 and p2[0] >= left_x_seg2) and (p2[1] <= right_y_seg2 and p2[1] >= left_y_seg2) and or2 == 0:
        return True

    if (p3[0] <= right_x_seg1 and p3[0] >= left_x_seg1) and (p3[1] <= right_y_seg1 and p3[1] >= left_y_seg1) and or3 == 0:
        return True

    if (p4[0] <= right_x_seg1 and p4[0] >= left_x_seg1) and (p4[1] <= right_y_seg1 and p4[1] >= left_y_seg1) and or4 == 0:
        return True

    return False
    return None

def segment_distance(segment1, segment2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.
    Hint: Distance of two line segments is the distance between the closest pair of points on both.

        Args:
            segment1: A tuple of coordinates indicating the endpoints of segment1.
            segment2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(segment1, segment2): return 0

    #else they are parallel so you can just calculate distance from endpoints

    d1 = point_segment_distance((segment1[0][0], segment1[0][1]), segment2)
    d2 = point_segment_distance((segment1[1][0], segment1[1][1]), segment2)
    d3 = point_segment_distance((segment2[0][0], segment2[0][1]), segment1)
    d4 = point_segment_distance((segment2[1][0], segment2[1][1]), segment1)

    return min(d1, d2, d3, d4)
    return -1

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result

    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                  f'{b} is expected to be {result[i]}, but your' \
                                                                  f'result is {distance}'

    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0)
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert touch_goal_result == truths[
            1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, ' \
                f'expected: {truths[1]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")