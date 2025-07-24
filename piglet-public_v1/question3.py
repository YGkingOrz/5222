
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
import heapq
from collections import defaultdict, deque
import random

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
    from flatland.core.grid.grid4_utils import get_new_position
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
# Optimized Multi-Agent Path Finding
#########################

def manhattan_distance(pos1, pos2):
    """安全的曼哈顿距离计算"""
    if pos1 is None or pos2 is None:
        return float('inf')
    try:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    except (TypeError, IndexError):
        return float('inf')

def is_valid_position(rail, pos):
    """检查位置是否有效"""
    if pos is None:
        return False
    try:
        return (0 <= pos[0] < rail.height and 0 <= pos[1] < rail.width and 
                rail.grid[pos[0]][pos[1]] > 0)
    except (TypeError, IndexError):
        return False

def get_safe_position(agent):
    """安全获取智能体位置"""
    if hasattr(agent, 'position') and agent.position is not None:
        return agent.position
    elif hasattr(agent, 'initial_position') and agent.initial_position is not None:
        return agent.initial_position
    else:
        return (0, 0)

def get_safe_target(agent):
    """安全获取智能体目标"""
    if hasattr(agent, 'target') and agent.target is not None:
        return agent.target
    else:
        return get_safe_position(agent)

def get_neighbors(rail, pos, direction):
    """获取有效的邻居位置"""
    if not is_valid_position(rail, pos):
        return []
    
    neighbors = []
    try:
        transitions = rail.get_transitions(pos[0], pos[1], direction)
        
        for new_dir, valid in enumerate(transitions):
            if valid:
                new_pos = get_new_position(pos, new_dir)
                if is_valid_position(rail, new_pos):
                    neighbors.append((new_pos, new_dir))
    except Exception:
        pass
    
    return neighbors

def time_sensitive_astar(start_pos, target_pos, rail, start_dir, occupied, max_time, deadline, current_time=0):
    if start_pos is None or target_pos is None or start_pos == target_pos:
        return [start_pos] if start_pos else [(0, 0)]
    if not is_valid_position(rail, start_pos):
        return [start_pos]
    max_nodes = 3000
    open_list = [(manhattan_distance(start_pos, target_pos) + max(0, current_time - deadline), 0, 0, start_pos, start_dir, [start_pos])]
    visited = set()
    consecutive_waits = 0
    max_consecutive_waits = 5  # 新增：限制连续等待
    while open_list:
        f_cost, g_cost, time_step, pos, direction, path = heapq.heappop(open_list)
        if len(path) > max_nodes:
            continue
        if pos == target_pos:
            if time_step <= deadline:
                return path
            # 允许轻微延迟但惩罚
            return path if g_cost < max_time else [start_pos]
        state = (pos, direction, time_step % 10)  # 周期避免visited爆炸
        if state in visited or time_step > max_time:
            continue
        visited.add(state)
        # 等待动作（成本1.5以更强惩罚）
        if consecutive_waits < max_consecutive_waits:
            wait_cost = g_cost + 1.5
            wait_time = time_step + 1
            wait_penalty = max(0, wait_time - deadline)
            h_cost = manhattan_distance(pos, target_pos) + wait_penalty
            if pos not in occupied.get(wait_time, set()):
                heapq.heappush(open_list, (wait_cost + h_cost, wait_cost, wait_time, pos, direction, path + [pos]))
                consecutive_waits += 1
            else:
                consecutive_waits = 0
        # 移动动作
        for new_pos, new_dir in get_neighbors(rail, pos, direction):
            move_time = time_step + 1
            if new_pos not in occupied.get(move_time, set()) and new_pos not in path[-3:]:
                new_g_cost = g_cost + 1
                move_penalty = max(0, move_time - deadline)
                h_cost = manhattan_distance(new_pos, target_pos) + move_penalty
                heapq.heappush(open_list, (new_g_cost + h_cost, new_g_cost, move_time, new_pos, new_dir, path + [new_pos]))
                consecutive_waits = 0
    return [start_pos]

def priority_planning(agents, rail, max_timestep):
    agent_count = len(agents)
    path_all = [[] for _ in range(agent_count)]
    # 复合优先级：deadline + 估计路径长度（紧迫+难度）
    agent_priorities = []
    for i, agent in enumerate(agents):
        deadline = agent.deadline if hasattr(agent, 'deadline') else max_timestep
        est_length = manhattan_distance(get_safe_position(agent), get_safe_target(agent))
        priority = deadline + est_length * 0.1 + random.uniform(0, 1)
        agent_priorities.append((i, priority))
    agent_priorities.sort(key=lambda x: x[1])
    # 时空占用
    occupied = defaultdict(set)
    for agent_idx, _ in agent_priorities:
        agent = agents[agent_idx]
        start_pos = get_safe_position(agent)
        target_pos = get_safe_target(agent)
        start_dir = getattr(agent, 'initial_direction', 0)
        deadline = agent.deadline if hasattr(agent, 'deadline') else max_timestep
        time_limit = min(50, max_timestep // 2)
        path = time_sensitive_astar(start_pos, target_pos, rail, start_dir, occupied, time_limit, deadline)
        path_all[agent_idx] = path
        # 更新占用
        for t, pos in enumerate(path):
            if pos:
                occupied[t].add(pos)
    return path_all

def extend_path_optimized(path, target, max_length, occupied, current_time):
    extended = path[:]
    current_pos = path[-1] if path else target
    for t in range(len(path), max_length):
        if current_pos == target:
            extended.append(target)
            continue
        # 尝试移动到目标，避免占用和死锁
        best_pos = current_pos
        min_dist = float('inf')
        for new_pos, _ in get_neighbors(global_rail, current_pos, 0):  # 简化方向
            if new_pos not in occupied.get(t + current_time, set()) and new_pos != extended[-2 if len(extended) > 1 else -1]:  # 避免立即返回
                dist = manhattan_distance(new_pos, target)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = new_pos
        if best_pos == current_pos and random.random() < 0.1:  # 随机打破潜在死锁
            best_pos = random.choice(get_neighbors(global_rail, current_pos, 0))[0] if get_neighbors(global_rail, current_pos, 0) else current_pos
        extended.append(best_pos)
        current_pos = best_pos
    return extended

def detect_enhanced_conflicts(paths, timestep, window=10):  # 扩大窗口
    conflicts = set()
    max_len = max(len(p) for p in paths) if paths else 0
    for t in range(max(0, timestep), min(timestep + window, max_len)):
        positions = defaultdict(list)
        for aid, path in enumerate(paths):
            if t < len(path) and path[t]:
                positions[path[t]].append(aid)
        for agents in positions.values():
            if len(agents) > 1:
                conflicts.update(agents)
        # 交换冲突
        if t > 0:
            for aid1, path1 in enumerate(paths):
                for aid2 in range(aid1 + 1, len(paths)):
                    path2 = paths[aid2]
                    if t < len(path1) and t < len(path2) and t-1 < len(path1) and t-1 < len(path2):
                        if path1[t] == path2[t-1] and path1[t-1] == path2[t]:
                            conflicts.update([aid1, aid2])
                        # 头对头
                        if path1[t] == path2[t] and path1[t-1] == path2[t-1] and path1[t] != path1[t-1]:
                            conflicts.update([aid1, aid2])
    # 简单死锁检测：代理循环
    for aid, path in enumerate(paths):
        if len(path) > 5 and path[-1] == path[-3] and path[-2] == path[-4]:
            conflicts.add(aid)
    return list(conflicts)

def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    global global_rail
    global_rail = rail
    random.seed(42)
    try:
        path_all = priority_planning(agents, rail, max_timestep)
        occupied = defaultdict(set)
        for t in range(max_timestep):
            for path in path_all:
                if t < len(path) and path[t]:
                    occupied[t].add(path[t])
        for i, agent in enumerate(agents):
            target = get_safe_target(agent)
            path_all[i] = extend_path_optimized(path_all[i], target, max_timestep, occupied, 0)
        return path_all
    except:
        return [[get_safe_position(a)] * max_timestep for a in agents]

def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, existing_paths: List[Tuple], max_timestep: int, new_malfunction_agents: List[int], failed_agents: List[int]):
    global global_rail
    global_rail = rail
    try:
        new_paths = [list(p) for p in existing_paths]
        agents_to_replan = set(new_malfunction_agents + failed_agents)
        max_iterations = 3  # 迭代重规划
        for _ in range(max_iterations):
            conflict_agents = detect_enhanced_conflicts(new_paths, current_timestep, window=10)
            agents_to_replan.update(conflict_agents)  # 无数量限制，但迭代控制
            occupied = defaultdict(set)
            for t in range(current_timestep, max_timestep):
                for aid, path in enumerate(new_paths):
                    if aid not in agents_to_replan and t < len(path) and path[t]:
                        occupied[t].add(path[t])
            for agent_id in list(agents_to_replan):
                if agent_id >= len(agents) or current_timestep >= max_timestep:
                    continue
                agent = agents[agent_id]
                current_pos = get_safe_position(agent)
                target_pos = get_safe_target(agent)
                direction = getattr(agent, 'direction', getattr(agent, 'initial_direction', 0))
                deadline = agent.deadline if hasattr(agent, 'deadline') else max_timestep
                remaining = max_timestep - current_timestep
                malfunction = agent.malfunction_data.get('malfunction', 0) if hasattr(agent, 'malfunction_data') else 0
                # 等待故障结束
                for wt in range(current_timestep, min(current_timestep + malfunction, max_timestep)):
                    if wt < len(new_paths[agent_id]):
                        new_paths[agent_id][wt] = current_pos
                    else:
                        new_paths[agent_id].append(current_pos)
                start_time = current_timestep + malfunction
                if start_time >= max_timestep:
                    continue
                new_segment = time_sensitive_astar(current_pos, target_pos, rail, direction, occupied, remaining - malfunction, deadline, start_time)
                if new_segment:
                    prefix = new_paths[agent_id][:start_time]
                    extended = extend_path_optimized(new_segment, target_pos, max_timestep - start_time, occupied, start_time)
                    new_paths[agent_id] = prefix + extended
                # 更新占用
                for t in range(start_time, max_timestep):
                    if t < len(new_paths[agent_id]) and new_paths[agent_id][t]:
                        occupied[t].add(new_paths[agent_id][t])
                # 长度调整
                if len(new_paths[agent_id]) > max_timestep:
                    new_paths[agent_id] = new_paths[agent_id][:max_timestep]
                else:
                    last = new_paths[agent_id][-1] if new_paths[agent_id] else current_pos
                    while len(new_paths[agent_id]) < max_timestep:
                        new_paths[agent_id].append(last)
            # 如果无新冲突，停止迭代
            if not detect_enhanced_conflicts(new_paths, current_timestep, window=10):
                break
        return new_paths
    except:
        return existing_paths

#####################################################################
# Instantiate a Remote Client
#####################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan)




