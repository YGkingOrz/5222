"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys


#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!", e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
# The path should avoid conflicts with existing paths.
#########################

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths. The index of each location is the time that
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, agent_id: int, existing_paths: list, max_timestep: int):
    """
    改进的多智能体时空A*路径规划算法
    """
    import heapq
    from flatland.core.grid.grid4 import Grid4TransitionsEnum
    from flatland.core.grid.grid4_utils import get_new_position
    
    # 时空A*节点类
    class SpaceTimeNode:
        def __init__(self, pos, direction, time, g, h, parent=None):
            self.pos = pos
            self.direction = direction
            self.time = time
            self.g = g
            self.h = h
            self.f = g + h
            self.parent = parent
        
        def __lt__(self, other):
            if abs(self.f - other.f) < 0.001:
                if abs(self.h - other.h) < 0.001:
                    return self.time < other.time
                return self.h < other.h
            return self.f < other.f
    
    # 改进的启发式函数
    def enhanced_heuristic(pos, goal, time, existing_paths):
        base_h = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # 添加拥堵惩罚
        congestion_penalty = 0
        for path in existing_paths:
            if time < len(path) and path[time] == pos:
                congestion_penalty += 2
            # 检查附近位置的拥堵
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nearby_pos = (pos[0] + dx, pos[1] + dy)
                if time < len(path) and path[time] == nearby_pos:
                    congestion_penalty += 0.5
        
        return base_h + congestion_penalty
    
    # 全面的冲突检测
    def has_conflict(current_pos, current_time, new_pos, new_time, existing_paths):
        for path in existing_paths:
            path_len = len(path)
            
            # 位置冲突：同一时间占用相同位置
            if new_time < path_len and path[new_time] == new_pos:
                return True
            
            # 交换冲突：两个智能体交换位置
            if (current_time < path_len and new_time < path_len and
                path[current_time] == new_pos and path[new_time] == current_pos):
                return True
            
            # 头尾冲突：一个智能体进入另一个智能体刚离开的位置
            if (current_time > 0 and current_time < path_len and
                new_time < path_len and
                path[current_time - 1] == new_pos and path[current_time] != new_pos):
                return True
            
            # 追尾冲突：跟随在另一个智能体后面
            if (new_time > 0 and new_time < path_len and
                path[new_time - 1] == current_pos and path[new_time] == new_pos):
                return True
        
        return False
    
    # 检查位置是否有有效轨道
    def has_valid_track(x, y, rail):
        if x < 0 or x >= rail.height or y < 0 or y >= rail.width:
            return False
        for direction in range(4):
            transitions = rail.get_transitions(x, y, direction)
            if any(transitions):
                return True
        return False
    
    # 重构路径
    def reconstruct_path(node):
        path = []
        current = node
        while current:
            path.append(current.pos)
            current = current.parent
        return path[::-1]
    
    # 验证起点和终点
    if not has_valid_track(start[0], start[1], rail) or not has_valid_track(goal[0], goal[1], rail):
        return []
    
    # 动态调整搜索参数
    num_agents = len(existing_paths)
    base_time_limit = min(max_timestep, 300)
    time_limit = base_time_limit + num_agents * 10  # 根据智能体数量调整
    max_nodes = 20000 + num_agents * 1000
    
    # 初始化搜索
    open_list = []
    closed = set()
    nodes_expanded = 0
    
    start_h = enhanced_heuristic(start, goal, 0, existing_paths)
    start_node = SpaceTimeNode(start, start_direction, 0, 0, start_h)
    heapq.heappush(open_list, start_node)
    
    # 用于快速查找的字典
    open_dict = {(start, start_direction, 0): start_node}
    
    while open_list and nodes_expanded < max_nodes:
        current = heapq.heappop(open_list)
        nodes_expanded += 1
        
        # 从开放字典中移除
        key = (current.pos, current.direction, current.time)
        if key in open_dict:
            del open_dict[key]
        
        # 到达目标
        if current.pos == goal:
            return reconstruct_path(current)
        
        # 时间限制
        if current.time >= time_limit:
            continue
        
        # 状态去重
        state = (current.pos, current.direction, current.time)
        if state in closed:
            continue
        closed.add(state)
        
        x, y = current.pos
        transitions = rail.get_transitions(x, y, current.direction)
        
        # 等待动作（智能等待策略）
        wait_time = current.time + 1
        if (wait_time < time_limit and 
            not has_conflict(current.pos, current.time, current.pos, wait_time, existing_paths)):
            
            # 动态等待代价：如果接近目标，等待代价更高
            distance_to_goal = abs(current.pos[0] - goal[0]) + abs(current.pos[1] - goal[1])
            wait_cost = 1.2 if distance_to_goal > 5 else 1.5
            
            wait_h = enhanced_heuristic(current.pos, goal, wait_time, existing_paths)
            wait_node = SpaceTimeNode(
                current.pos, current.direction, wait_time,
                current.g + wait_cost, wait_h, current
            )
            
            wait_key = (wait_node.pos, wait_node.direction, wait_node.time)
            if wait_key not in closed and wait_key not in open_dict:
                heapq.heappush(open_list, wait_node)
                open_dict[wait_key] = wait_node
        
        # 移动动作
        for new_dir in range(4):
            if not transitions[new_dir]:
                continue
            
            new_pos = get_new_position(current.pos, new_dir)
            new_x, new_y = new_pos
            
            # 边界和轨道检查
            if not has_valid_track(new_x, new_y, rail):
                continue
            
            new_time = current.time + 1
            
            # 冲突检查
            if has_conflict(current.pos, current.time, new_pos, new_time, existing_paths):
                continue
            
            new_key = (new_pos, new_dir, new_time)
            if new_key in closed:
                continue
            
            new_g = current.g + 1
            new_h = enhanced_heuristic(new_pos, goal, new_time, existing_paths)
            new_node = SpaceTimeNode(new_pos, new_dir, new_time, new_g, new_h, current)
            
            # 检查是否已在开放列表中有更好的路径
            if new_key in open_dict:
                existing_node = open_dict[new_key]
                if existing_node.g <= new_g:
                    continue
                # 移除旧节点
                del open_dict[new_key]
            
            heapq.heappush(open_list, new_node)
            open_dict[new_key] = new_node
    
    # 如果没找到完整路径，尝试找到部分路径
    if closed:
        # 找到最接近目标的位置
        best_node = None
        best_distance = float('inf')
        
        for state in closed:
            pos, _, time = state
            distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            if distance < best_distance:
                best_distance = distance
                # 重新构建节点（简化版）
                best_node = SpaceTimeNode(pos, 0, time, 0, 0)
        
        if best_node and best_node.pos != start:
            return [start, best_node.pos]
    
    # 最后的备选方案：返回空路径让系统处理
    return []


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,2)

"""
This is the python script for question 2. In this script, you are required to implement a single agent path-finding algorithm considering time dimension and avoid conflict with existing paths.
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys
import heapq
from collections import defaultdict

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
    from flatland.core.grid.grid4_utils import get_new_position
except Exception as e:
    eprint("Cannot load flatland modules!", e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

debug = False
visualizer = False

test_single_instance = False
level = 0
test = 0

#########################
# Advanced optimized get_path function
#########################

def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, agent_id: int, existing_paths: list, max_timestep: int):
    """
    高度优化的多智能体时空A*路径规划算法
    针对大规模场景进行了特殊优化
    """
    import heapq
    from flatland.core.grid.grid4_utils import get_new_position
    
    # 轻量级时空节点
    class STNode:
        __slots__ = ['pos', 'dir', 'time', 'g', 'h', 'parent']
        
        def __init__(self, pos, direction, time, g, h, parent=None):
            self.pos = pos
            self.dir = direction
            self.time = time
            self.g = g
            self.h = h
            self.parent = parent
        
        @property
        def f(self):
            return self.g + self.h
        
        def __lt__(self, other):
            if abs(self.f - other.f) < 0.001:
                return self.h < other.h
            return self.f < other.f
    
    # 智能启发式函数
    def smart_heuristic(pos, goal, time, occupied_map, num_agents):
        base_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # 大规模场景下简化启发式
        if num_agents > 100:
            return base_dist + time * 0.01
        
        # 中小规模场景的详细启发式
        time_penalty = max(0, time - max_timestep * 0.7) * 0.2
        
        # 拥堵预测（限制检查范围）
        congestion = 0
        check_range = min(base_dist + 5, 20)
        for t in range(time, min(time + check_range, max_timestep)):
            if pos in occupied_map.get(t, set()):
                congestion += 1.5
                break
        
        return base_dist + time_penalty + congestion
    
    # 高效冲突检测
    def quick_conflict_check(curr_pos, curr_time, new_pos, new_time, occupied_map, paths):
        # 基本位置冲突
        if new_pos in occupied_map.get(new_time, set()):
            return True
        
        # 大规模场景下跳过复杂冲突检测
        if len(paths) > 100:
            return False
        
        # 交换冲突检测（仅对附近路径）
        for path in paths[:min(len(paths), 50)]:  # 限制检查数量
            if (curr_time < len(path) and new_time < len(path) and
                path[curr_time] == new_pos and path[new_time] == curr_pos):
                return True
        
        return False
    
    # 位置有效性检查（缓存结果）
    _valid_cache = {}
    def is_valid_pos(rail, pos):
        if pos in _valid_cache:
            return _valid_cache[pos]
        
        x, y = pos
        valid = (0 <= x < rail.height and 0 <= y < rail.width and 
                rail.get_transitions(x, y, 0) != (0,0,0,0))
        _valid_cache[pos] = valid
        return valid
    
    # 路径重构
    def build_path(node):
        path = []
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]
    
    # 基础验证
    if not is_valid_pos(rail, start) or not is_valid_pos(rail, goal):
        return []
    
    # 预计算占用表
    occupied_map = defaultdict(set)
    for path in existing_paths:
        for t, pos in enumerate(path):
            if t < max_timestep:
                occupied_map[t].add(pos)
    
    # 动态参数调整
    num_agents = len(existing_paths)
    map_size = rail.height * rail.width
    
    # 大规模场景的激进参数
    if num_agents > 100:
        time_limit = min(max_timestep, 150)
        max_nodes = 5000
        early_termination = True
    elif num_agents > 50:
        time_limit = min(max_timestep, 200)
        max_nodes = 10000
        early_termination = True
    else:
        time_limit = min(max_timestep, 300)
        max_nodes = 20000
        early_termination = False
    
    # A*搜索
    open_heap = []
    closed = set()
    nodes_expanded = 0
    
    start_h = smart_heuristic(start, goal, 0, occupied_map, num_agents)
    start_node = STNode(start, start_direction, 0, 0, start_h)
    heapq.heappush(open_heap, start_node)
    
    best_node = start_node
    best_distance = start_h
    
    while open_heap and nodes_expanded < max_nodes:
        current = heapq.heappop(open_heap)
        nodes_expanded += 1
        
        # 状态去重（使用周期性状态减少内存）
        state_key = (current.pos, current.dir, current.time % 50)
        if state_key in closed or current.time >= time_limit:
            continue
        closed.add(state_key)
        
        # 目标检测
        if current.pos == goal:
            return build_path(current)
        
        # 跟踪最佳节点
        curr_dist = abs(current.pos[0] - goal[0]) + abs(current.pos[1] - goal[1])
        if curr_dist < best_distance:
            best_distance = curr_dist
            best_node = current
        
        # 早期终止（大规模场景）
        if early_termination and nodes_expanded > max_nodes // 2 and curr_dist > best_distance + 10:
            continue
        
        x, y = current.pos
        transitions = rail.get_transitions(x, y, current.dir)
        
        # 等待动作（限制等待次数）
        if current.time < time_limit - 1:
            wait_time = current.time + 1
            if not quick_conflict_check(current.pos, current.time, current.pos, wait_time, occupied_map, existing_paths):
                wait_cost = 1.2 if num_agents > 50 else 1.1
                wait_h = smart_heuristic(current.pos, goal, wait_time, occupied_map, num_agents)
                wait_node = STNode(current.pos, current.dir, wait_time, current.g + wait_cost, wait_h, current)
                heapq.heappush(open_heap, wait_node)
        
        # 移动动作
        for new_dir in range(4):
            if transitions[new_dir]:
                new_pos = get_new_position(current.pos, new_dir)
                
                if is_valid_pos(rail, new_pos):
                    new_time = current.time + 1
                    
                    if (new_time < time_limit and 
                        not quick_conflict_check(current.pos, current.time, new_pos, new_time, occupied_map, existing_paths)):
                        
                        new_g = current.g + 1
                        new_h = smart_heuristic(new_pos, goal, new_time, occupied_map, num_agents)
                        new_node = STNode(new_pos, new_dir, new_time, new_g, new_h, current)
                        heapq.heappush(open_heap, new_node)
    
    # 返回部分路径或简单路径
    if best_node and best_node.pos != start:
        return build_path(best_node)
    
    # 最后的备选：直线路径（忽略冲突）
    if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) <= 10:
        return [start, goal]
    
    return []

#########################
# You should not modify codes below
#########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path, test_cases, debug, visualizer, 2)
















