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
    eprint("Cannot load flatland modules!")
    eprint(e)
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
#########################


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
    """
    使用改进的A*算法实现单智能体路径查找
    """
    import heapq
    from flatland.core.grid.grid4 import Grid4TransitionsEnum
    from flatland.core.grid.grid4_utils import get_new_position
    
    # A*节点类
    class AStarNode:
        def __init__(self, position, direction, g_cost=0, h_cost=0, parent=None):
            self.position = position  # (x, y)
            self.direction = direction  # 当前朝向
            self.g_cost = g_cost  # 从起点到当前节点的实际代价
            self.h_cost = h_cost  # 启发式代价（曼哈顿距离）
            self.f_cost = g_cost + h_cost  # 总代价
            self.parent = parent
        
        def __lt__(self, other):
            if self.f_cost == other.f_cost:
                return self.h_cost < other.h_cost  # 优先选择更接近目标的节点
            return self.f_cost < other.f_cost
    
    # 计算曼哈顿距离启发式函数
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # 重构路径
    def reconstruct_path(node):
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # 反转路径
    
    # 检查位置是否有效
    def is_valid_position(x, y):
        return 0 <= x < rail.height and 0 <= y < rail.width
    
    # 检查位置是否有轨道
    def has_track(x, y):
        if not is_valid_position(x, y):
            return False
        # 检查该位置是否有任何方向的轨道连接
        for direction in range(4):
            transitions = rail.get_transitions(x, y, direction)
            if any(transitions):
                return True
        return False
    
    # 验证起点和终点
    if not has_track(start[0], start[1]) or not has_track(goal[0], goal[1]):
        return []
    
    # 初始化开放列表和关闭列表
    open_list = []
    closed_set = set()
    # 使用字典来快速查找开放列表中的节点
    open_dict = {}
    
    # 创建起始节点
    start_node = AStarNode(
        position=start,
        direction=start_direction,
        g_cost=0,
        h_cost=manhattan_distance(start, goal)
    )
    
    heapq.heappush(open_list, start_node)
    open_dict[(start, start_direction)] = start_node
    
    # A*主循环
    while open_list:
        # 获取f值最小的节点
        current_node = heapq.heappop(open_list)
        
        # 从开放字典中移除
        key = (current_node.position, current_node.direction)
        if key in open_dict:
            del open_dict[key]
        
        # 如果到达目标，检查是否可以在目标位置停止
        if current_node.position == goal:
            # 验证目标位置的轨道连接
            x, y = goal
            goal_transitions = rail.get_transitions(x, y, current_node.direction)
            # 如果目标位置有有效的轨道连接，返回路径
            if any(goal_transitions) or current_node.position == start:
                return reconstruct_path(current_node)
        
        # 将当前节点加入关闭列表
        closed_set.add((current_node.position, current_node.direction))
        
        # 获取当前位置的可用转换
        x, y = current_node.position
        valid_transitions = rail.get_transitions(x, y, current_node.direction)
        
        # 遍历所有可能的方向
        for new_direction in range(4):  # NORTH=0, EAST=1, SOUTH=2, WEST=3
            # 检查是否可以朝这个方向移动
            if not valid_transitions[new_direction]:
                continue
            
            # 计算新位置
            new_position = get_new_position(current_node.position, new_direction)
            new_x, new_y = new_position
            
            # 修复边界检查错误
            if not is_valid_position(new_x, new_y):
                continue
            
            # 检查新位置是否有轨道
            if not has_track(new_x, new_y):
                continue
            
            # 检查是否已经在关闭列表中
            if (new_position, new_direction) in closed_set:
                continue
            
            # 计算新节点的代价
            new_g_cost = current_node.g_cost + 1
            new_h_cost = manhattan_distance(new_position, goal)
            
            # 检查是否已经在开放列表中
            open_key = (new_position, new_direction)
            if open_key in open_dict:
                existing_node = open_dict[open_key]
                if existing_node.g_cost <= new_g_cost:
                    continue  # 已有更好的路径
                # 移除旧节点（虽然还在堆中，但会被忽略）
                del open_dict[open_key]
            
            # 创建新节点
            new_node = AStarNode(
                position=new_position,
                direction=new_direction,
                g_cost=new_g_cost,
                h_cost=new_h_cost,
                parent=current_node
            )
            
            # 添加到开放列表
            heapq.heappush(open_list, new_node)
            open_dict[open_key] = new_node
        
        # 防止无限循环，如果搜索时间过长则退出
        if current_node.g_cost > max_timestep:
            break
    
    # 如果没有找到路径，返回空列表
    return []


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,1)



















