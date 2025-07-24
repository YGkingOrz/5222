
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
import random
import heapq
from collections import defaultdict, deque

#import necessary modules that this python scripts need.
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
# Global variables for LNS algorithm
#########################

# Neighborhood weights for adaptive selection
neighborhood_weights = {
    'collision': 1.0,
    'failure': 1.0, 
    'random': 1.0
}

# Success tracking for neighborhoods
neighborhood_success = {
    'collision': 0,
    'failure': 0,
    'random': 0
}

class LNSPathPlanner:
    def __init__(self, rail: GridTransitionMap, agents: List[EnvAgent], max_timestep: int):
        self.rail = rail
        self.agents = agents
        self.max_timestep = max_timestep
        self.paths = [[] for _ in range(len(agents))]
        self.conflict_graph = defaultdict(set)
        
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_valid_moves(self, pos, direction):
        """Get valid moves from current position and direction"""
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
        
        return moves
    
    def a_star_single_agent(self, agent_id, existing_paths, start_time=0):
        """A* search for single agent avoiding existing paths"""
        agent = self.agents[agent_id]
        start_pos = agent.initial_position
        goal_pos = agent.target
        start_dir = agent.initial_direction
        
        # Priority queue: (f_cost, g_cost, time, pos, direction, path)
        open_set = [(0, 0, start_time, start_pos, start_dir, [start_pos])]
        closed_set = set()
        
        while open_set:
            f_cost, g_cost, current_time, pos, direction, path = heapq.heappop(open_set)
            
            if (current_time, pos, direction) in closed_set:
                continue
                
            closed_set.add((current_time, pos, direction))
            
            # Check if reached goal
            if pos == goal_pos:
                # Extend path to max_timestep if needed
                while len(path) < self.max_timestep and current_time < self.max_timestep:
                    path.append(goal_pos)
                    current_time += 1
                return path
            
            # If time limit exceeded
            if current_time >= self.max_timestep - 1:
                continue
            
            # Try all valid moves
            valid_moves = self.get_valid_moves(pos, direction)
            
            # Add wait action
            valid_moves.append((pos, direction))
            
            for new_pos, new_dir in valid_moves:
                new_time = current_time + 1
                
                # Check conflicts with existing paths
                conflict = False
                for other_agent_id, other_path in enumerate(existing_paths):
                    if other_agent_id == agent_id or not other_path:
                        continue
                    
                    # Vertex conflict
                    if new_time < len(other_path) and other_path[new_time] == new_pos:
                        conflict = True
                        break
                    
                    # Edge conflict
                    if (new_time < len(other_path) and 
                        new_time - 1 >= 0 and new_time - 1 < len(other_path) and
                        other_path[new_time] == pos and other_path[new_time - 1] == new_pos):
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                if (new_time, new_pos, new_dir) in closed_set:
                    continue
                
                new_g = g_cost + 1
                new_h = self.manhattan_distance(new_pos, goal_pos)
                new_f = new_g + new_h
                new_path = path + [new_pos]
                
                heapq.heappush(open_set, (new_f, new_g, new_time, new_pos, new_dir, new_path))
        
        return []  # No path found
    
    def count_conflicts(self, paths):
        """Count total number of conflicts in current paths"""
        conflicts = 0
        for t in range(min(len(p) for p in paths if p)):
            positions = []
            for path in paths:
                if t < len(path):
                    positions.append(path[t])
            
            # Count vertex conflicts
            position_count = defaultdict(int)
            for pos in positions:
                position_count[pos] += 1
                if position_count[pos] > 1:
                    conflicts += 1
        
        return conflicts
    
    def build_conflict_graph(self, paths):
        """Build conflict graph between agents"""
        self.conflict_graph.clear()
        
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if not paths[i] or not paths[j]:
                    continue
                
                # Check for conflicts between agent i and j
                max_time = min(len(paths[i]), len(paths[j]))
                for t in range(max_time):
                    # Vertex conflict
                    if paths[i][t] == paths[j][t]:
                        self.conflict_graph[i].add(j)
                        self.conflict_graph[j].add(i)
                        break
                    
                    # Edge conflict
                    if (t > 0 and 
                        paths[i][t] == paths[j][t-1] and 
                        paths[i][t-1] == paths[j][t]):
                        self.conflict_graph[i].add(j)
                        self.conflict_graph[j].add(i)
                        break
    
    def select_neighborhood(self, paths, neighborhood_type, k=5):
        """Select agents for destroy phase based on neighborhood type"""
        if neighborhood_type == 'collision':
            return self.collision_neighborhood(paths, k)
        elif neighborhood_type == 'failure':
            return self.failure_neighborhood(paths, k)
        else:  # random
            return self.random_neighborhood(k)
    
    def collision_neighborhood(self, paths, k):
        """Select agents based on collision graph"""
        self.build_conflict_graph(paths)
        
        if not self.conflict_graph:
            return self.random_neighborhood(k)
        
        # Find connected components
        visited = set()
        components = []
        
        for agent_id in self.conflict_graph:
            if agent_id not in visited:
                component = set()
                queue = deque([agent_id])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        for neighbor in self.conflict_graph[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(component)
        
        if components:
            # Select from largest component
            largest_component = max(components, key=len)
            return list(largest_component)[:k]
        
        return self.random_neighborhood(k)
    
    def failure_neighborhood(self, paths, k):
        """Select agents that prevent others from completing collision-free"""
        failed_agents = []
        
        for agent_id in range(len(self.agents)):
            if not paths[agent_id] or paths[agent_id][-1] != self.agents[agent_id].target:
                failed_agents.append(agent_id)
        
        if failed_agents:
            return failed_agents[:k]
        
        return self.collision_neighborhood(paths, k)
    
    def random_neighborhood(self, k):
        """Select k random agents"""
        return random.sample(range(len(self.agents)), min(k, len(self.agents)))
    
    def adaptive_neighborhood_selection(self):
        """Select neighborhood type based on adaptive weights"""
        total_weight = sum(neighborhood_weights.values())
        rand_val = random.random() * total_weight
        
        cumulative = 0
        for neighborhood_type, weight in neighborhood_weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                return neighborhood_type
        
        return 'random'
    
    def update_neighborhood_weights(self, neighborhood_type, improvement):
        """Update neighborhood weights based on success"""
        if improvement > 0:
            neighborhood_success[neighborhood_type] += 1
            neighborhood_weights[neighborhood_type] *= 1.1
        else:
            neighborhood_weights[neighborhood_type] *= 0.9
        
        # Normalize weights
        total_weight = sum(neighborhood_weights.values())
        for key in neighborhood_weights:
            neighborhood_weights[key] /= total_weight
    
    def prioritized_planning_init(self):
        """Initialize paths using prioritized planning"""
        agent_order = list(range(len(self.agents)))
        random.shuffle(agent_order)
        
        paths = [[] for _ in range(len(self.agents))]
        
        for agent_id in agent_order:
            path = self.a_star_single_agent(agent_id, paths)
            paths[agent_id] = path
        
        return paths
    
    def repair_paths(self, selected_agents, current_paths):
        """Repair paths for selected agents"""
        # Create a copy of current paths
        new_paths = [path[:] for path in current_paths]
        
        # Clear paths for selected agents
        for agent_id in selected_agents:
            new_paths[agent_id] = []
        
        # Re-plan for selected agents in random order
        random.shuffle(selected_agents)
        
        for agent_id in selected_agents:
            path = self.a_star_single_agent(agent_id, new_paths)
            new_paths[agent_id] = path
        
        return new_paths
    
    def lns_solve(self, time_limit=30, max_iterations=100):
        """Main LNS algorithm"""
        start_time = time.time()
        
        # Initialize with prioritized planning
        current_paths = self.prioritized_planning_init()
        current_conflicts = self.count_conflicts(current_paths)
        best_paths = [path[:] for path in current_paths]
        best_conflicts = current_conflicts
        
        iteration = 0
        no_improvement_count = 0
        
        while (time.time() - start_time < time_limit and 
               iteration < max_iterations and 
               no_improvement_count < 20):
            
            # Select neighborhood adaptively
            neighborhood_type = self.adaptive_neighborhood_selection()
            
            # Destroy: select agents to replan
            k = min(5 + iteration // 10, len(self.agents) // 2)  # Adaptive neighborhood size
            selected_agents = self.select_neighborhood(current_paths, neighborhood_type, k)
            
            if not selected_agents:
                no_improvement_count += 1
                iteration += 1
                continue
            
            # Repair: replan selected agents
            new_paths = self.repair_paths(selected_agents, current_paths)
            new_conflicts = self.count_conflicts(new_paths)
            
            # Accept improvement or with probability for diversification
            improvement = current_conflicts - new_conflicts
            accept_probability = 0.1 if improvement <= 0 else 1.0
            
            if improvement > 0 or random.random() < accept_probability:
                current_paths = new_paths
                current_conflicts = new_conflicts
                
                if new_conflicts < best_conflicts:
                    best_paths = [path[:] for path in new_paths]
                    best_conflicts = new_conflicts
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Update neighborhood weights
            self.update_neighborhood_weights(neighborhood_type, improvement)
            
            iteration += 1
            
            # Early termination if no conflicts
            if best_conflicts == 0:
                break
        
        return best_paths

# Global planner instance
planner = None

def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """Main path planning function using LNS algorithm"""
    global planner
    
    # Create LNS planner
    planner = LNSPathPlanner(rail, agents, max_timestep)
    
    # Calculate time limit based on problem size
    time_limit = min(30, max(5, len(agents) * 0.5))
    
    # Solve using LNS
    paths = planner.lns_solve(time_limit=time_limit)
    
    # Ensure all paths are valid length
    for i, path in enumerate(paths):
        if not path:
            # If no path found, create a path that stays at initial position
            path = [agents[i].initial_position] * max_timestep
            paths[i] = path
        elif len(path) < max_timestep:
            # Extend path by staying at goal
            last_pos = path[-1] if path else agents[i].initial_position
            while len(path) < max_timestep:
                path.append(last_pos)
    
    return paths

def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, 
           existing_paths: List[Tuple], max_timestep: int, 
           new_malfunction_agents: List[int], failed_agents: List[int]):
    """Replan function to handle malfunctions and failures"""
    global planner
    
    if planner is None:
        planner = LNSPathPlanner(rail, agents, max_timestep)
    
    # Create new paths starting from current timestep
    new_paths = [list(path) for path in existing_paths]
    
    # Identify agents that need replanning
    agents_to_replan = set(new_malfunction_agents + failed_agents)
    
    # Add agents that are in conflict with failed/malfunctioned agents
    for agent_id in list(agents_to_replan):
        for other_id in range(len(agents)):
            if other_id not in agents_to_replan and other_id < len(existing_paths):
                # Check if there will be conflicts
                if (current_timestep < len(existing_paths[agent_id]) and 
                    current_timestep < len(existing_paths[other_id]) and
                    existing_paths[agent_id][current_timestep] == existing_paths[other_id][current_timestep]):
                    agents_to_replan.add(other_id)
    
    # Clear future paths for agents that need replanning
    for agent_id in agents_to_replan:
        if agent_id < len(new_paths):
            # Keep path up to current timestep, clear the rest
            new_paths[agent_id] = new_paths[agent_id][:current_timestep + 1]
    
    # Use LNS to replan
    if agents_to_replan:
        # Create temporary planner for replanning
        temp_planner = LNSPathPlanner(rail, agents, max_timestep)
        
        # Set current positions
        for agent_id in range(len(agents)):
            if current_timestep < len(new_paths[agent_id]):
                agents[agent_id].position = new_paths[agent_id][current_timestep]
        
        # Replan with shorter time limit
        time_limit = min(10, max(2, len(agents_to_replan) * 0.3))
        replanned_paths = temp_planner.repair_paths(list(agents_to_replan), new_paths)
        
        # Update paths
        for agent_id in agents_to_replan:
            if agent_id < len(replanned_paths) and replanned_paths[agent_id]:
                new_paths[agent_id] = replanned_paths[agent_id]
    
    # Ensure all paths have correct length
    for i, path in enumerate(new_paths):
        if len(path) < max_timestep:
            last_pos = path[-1] if path else agents[i].initial_position
            while len(path) < max_timestep:
                path.append(last_pos)
    
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




