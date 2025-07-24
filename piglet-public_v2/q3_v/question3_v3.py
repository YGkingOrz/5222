
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
import heapq
from collections import defaultdict, deque

try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
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
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################


# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
class MAPFSolver:
    def __init__(self, rail: GridTransitionMap, agents: List[EnvAgent], max_timestep: int):
        self.rail = rail
        self.agents = agents
        self.max_timestep = max_timestep
        self.paths = [[] for _ in range(len(agents))]
        
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_valid_moves(self, pos, direction):
        """获取当前位置和方向下的有效移动"""
        valid_transitions = self.rail.get_transitions(pos[0], pos[1], direction)
        moves = []
        
        for i, valid in enumerate(valid_transitions):
            if valid:
                new_x, new_y = pos[0], pos[1]
                if i == Directions.NORTH:
                    new_x -= 1
                elif i == Directions.EAST:
                    new_y += 1
                elif i == Directions.SOUTH:
                    new_x += 1
                elif i == Directions.WEST:
                    new_y -= 1
                
                if 0 <= new_x < self.rail.height and 0 <= new_y < self.rail.width:
                    moves.append(((new_x, new_y), i))
        
        # 添加等待动作
        moves.append((pos, direction))
        return moves
    
    def is_conflict(self, agent1_pos, agent2_pos, agent1_next, agent2_next):
        """检查两个智能体是否冲突"""
        # 位置冲突
        if agent1_next == agent2_next:
            return True
        # 交换冲突
        if agent1_pos == agent2_next and agent2_pos == agent1_next:
            return True
        return False
    
    def a_star_with_reservations(self, agent_id, reservations, start_time=0):
        """使用A*算法为单个智能体规划路径，考虑其他智能体的预留"""
        agent = self.agents[agent_id]
        start_pos = agent.initial_position
        start_dir = agent.initial_direction
        goal_pos = agent.target
        
        # 优先队列：(f_cost, g_cost, timestep, pos, direction, path)
        open_list = [(self.manhattan_distance(start_pos, goal_pos), 0, start_time, start_pos, start_dir, [start_pos])]
        closed_set = set()
        
        while open_list:
            f_cost, g_cost, timestep, pos, direction, path = heapq.heappop(open_list)
            
            if (timestep, pos, direction) in closed_set:
                continue
            closed_set.add((timestep, pos, direction))
            
            # 到达目标
            if pos == goal_pos:
                return path
            
            # 时间限制
            if timestep >= self.max_timestep:
                continue
            
            # 扩展邻居
            for next_pos, next_dir in self.get_valid_moves(pos, direction):
                next_timestep = timestep + 1
                
                # 检查预留冲突
                if (next_timestep, next_pos) in reservations:
                    continue
                
                # 检查交换冲突
                conflict = False
                for other_agent_id, other_path in enumerate(self.paths):
                    if other_agent_id != agent_id and next_timestep < len(other_path):
                        if self.is_conflict(pos, other_path[timestep] if timestep < len(other_path) else other_path[-1],
                                           next_pos, other_path[next_timestep]):
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                new_g = g_cost + 1
                new_h = self.manhattan_distance(next_pos, goal_pos)
                new_f = new_g + new_h
                new_path = path + [next_pos]
                
                heapq.heappush(open_list, (new_f, new_g, next_timestep, next_pos, next_dir, new_path))
        
        return []  # 无解
    
    def solve_prioritized(self):
        """使用优先级规划解决MAPF"""
        # 按照到目标的距离排序智能体优先级
        agent_priorities = sorted(range(len(self.agents)), 
                                key=lambda i: self.manhattan_distance(self.agents[i].initial_position, self.agents[i].target))
        
        reservations = set()
        
        for agent_id in agent_priorities:
            path = self.a_star_with_reservations(agent_id, reservations)
            if path:
                self.paths[agent_id] = path
                # 添加路径到预留表
                for t, pos in enumerate(path):
                    reservations.add((t, pos))
            else:
                # 如果找不到路径，使用简单的等待策略
                self.paths[agent_id] = [self.agents[agent_id].initial_position]
        
        return self.paths

def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """主要路径规划函数"""
    if not agents:
        return []
    
    solver = MAPFSolver(rail, agents, max_timestep)
    paths = solver.solve_prioritized()
    
    return paths

def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, 
          existing_paths: List[Tuple], max_timestep: int, 
          new_malfunction_agents: List[int], failed_agents: List[int]):
    """重规划函数处理故障和失败"""
    if debug:
        print(f"Replan called at timestep {current_timestep}", file=sys.stderr)
        print(f"New malfunction agents: {new_malfunction_agents}", file=sys.stderr)
        print(f"Failed agents: {failed_agents}", file=sys.stderr)
    
    # 复制现有路径
    new_paths = [list(path) for path in existing_paths]
    
    # 需要重新规划的智能体
    agents_to_replan = set(new_malfunction_agents + failed_agents)
    
    # 为受影响的智能体重新规划
    solver = MAPFSolver(rail, agents, max_timestep)
    solver.paths = new_paths
    
    for agent_id in agents_to_replan:
        agent = agents[agent_id]
        
        # 处理故障智能体
        if agent_id in new_malfunction_agents:
            malfunction_duration = agent.malfunction_data.get('malfunction', 0)
            current_pos = agent.position if agent.position else agent.initial_position
            
            # 在故障期间保持当前位置
            for t in range(current_timestep, min(current_timestep + malfunction_duration, max_timestep)):
                if t < len(new_paths[agent_id]):
                    new_paths[agent_id][t] = current_pos
                else:
                    new_paths[agent_id].append(current_pos)
        
        # 从当前时间步重新规划路径
        if current_timestep < len(new_paths[agent_id]):
            # 截断现有路径
            new_paths[agent_id] = new_paths[agent_id][:current_timestep + 1]
            
            # 重新规划剩余路径
            reservations = set()
            for other_id, other_path in enumerate(new_paths):
                if other_id != agent_id:
                    for t, pos in enumerate(other_path[current_timestep:], current_timestep):
                        reservations.add((t, pos))
            
            remaining_path = solver.a_star_with_reservations(agent_id, reservations, current_timestep)
            if remaining_path and len(remaining_path) > 1:
                new_paths[agent_id].extend(remaining_path[1:])  # 跳过当前位置
    
    return new_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)




