import networkx as nx
import numpy as np
from collections import defaultdict
import yaml
import matplotlib.pyplot as plt

class TrajectoryProcessor:
    def __init__(self, constraint_file, graph_constraints):
        self.hint_constraints = self.load_constraints(constraint_file)  # All constraints
        self.graph_constraints = graph_constraints  # Constraints that define graph vertices
        self.state_occurrences = {}

    def load_constraints(self, filepath):
        """Load all constraints from the YAML file."""
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            return data['hint_constraints']

    def store_trajectory(self, trajectory):
        """Store trajectories by extracting constraints that define graph vertices."""
        
        # print("\n=== DEBUG: store_trajectory ===")
        # print(f"[DEBUG] Storing Trajectory of Length: {len(trajectory)}")

        
        for t in trajectory:
            # Extract values directly from constraint dictionaries
            constraint_values = list(t.values())  
            # print(f"\nRaw Extracted Constraint Dictionary: {t}")
            # print(f"Extracted Constraint Values: {constraint_values}")

            # Convert to binary representation (0 or 1)
            binary_constraints = [1 if value > 0.5 else 0 for value in constraint_values]
            # print(f"Binary Constraints: {binary_constraints}")

            # Map constraints to their binary values
            full_symbolic_state = {key: bool(value) for key, value in zip(self.hint_constraints, binary_constraints)}
            # print(f"[DEBUG] Full Symbolic State: {full_symbolic_state}")

            # Filter for only graph-relevant constraints
            graph_state = {key: full_symbolic_state[key] for key in self.graph_constraints if key in full_symbolic_state}
            graph_state_tuple = tuple(sorted(graph_state.items()))  # Ensure consistent order

            # print(f"[DEBUG] Graph State Tuple: {graph_state_tuple}")

            # Track occurrences
            if graph_state_tuple not in self.state_occurrences:
                self.state_occurrences[graph_state_tuple] = 0
            self.state_occurrences[graph_state_tuple] += 1

            # print("=========================================") 

    def extract_constraints(self, state):
        """Extract only graph-related constraints from the state vector."""
        encoded_constraints = state[-len(self.hint_constraints):]  # Extract last N values
        
        # print(f"Expected constraint length: {len(self.hint_constraints)}, Actual state length: {len(state)}")
        # print(f"Extracted Encoded Constraints (Raw from State Vector): {encoded_constraints}")
        
        # Convert raw values to binary interpretation based on a threshold (e.g., 0.5)
        binary_constraints = [1 if value > 0 else 0 for value in encoded_constraints]

        full_symbolic_state = {key: bool(value) for key, value in zip(self.hint_constraints, binary_constraints)}
        
        # print(f"Interpreted Binary Constraints: {binary_constraints}")

        graph_state = {key: full_symbolic_state[key] for key in self.graph_constraints if key in full_symbolic_state}
        
        # print(f"Graph State: {graph_state}")

        return graph_state  # Return dictionary of relevant constraints

class TransitionGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.constraint_order = []  # Store constraint order for legend

    def add_trajectory(self, trajectory):
        """Add state transitions to the graph when at least one constraint changes."""
        prev_state = None
        # print(f"[DEBUG] Adding Trajectory with {len(trajectory)} states")

        for state in trajectory:
            state_tuple = tuple(sorted(state.items()))
            # print(f"[DEBUG] Processing State: {state_tuple}")


            if not self.constraint_order:
                self.constraint_order = [key for key, _ in state_tuple]  # Capture order once

            # Ensure the state node exists in the graph
            if state_tuple not in self.graph:
                self.graph.add_node(state_tuple)

            if prev_state:
                changed_constraints = [key for key in state if state[key] != prev_state[key]]

                if len(changed_constraints) >= 1:
                    transition_from = tuple(sorted(prev_state.items()))
                    transition_to = tuple(sorted(state.items()))
                    # print(f"[DEBUG] Transition: {transition_from} → {transition_to}")


                    if transition_from not in self.graph:
                        self.graph.add_node(transition_from)
                    if transition_to not in self.graph:
                        self.graph.add_node(transition_to)

                    # Add an edge
                    if self.graph.has_edge(transition_from, transition_to):
                        self.graph[transition_from][transition_to]["weight"] += 1
                    else:
                        self.graph.add_edge(transition_from, transition_to, weight=1)

                    # print(f"[DEBUG] Edge Added: {transition_from} → {transition_to}")

            prev_state = state

    def visualize_graph(self, save_path=None):
        """Plot the transition graph with binary node labels and an explicit legend."""
        pos = nx.spring_layout(self.graph)  # Position nodes

        # Convert full constraint dict to binary string
        def encode_binary_label(state_tuple):
            """Convert state tuples into compact binary string representation."""
            binary_vector = "".join(["1" if val else "0" for _, val in state_tuple])
            return binary_vector

        node_labels = {node: encode_binary_label(node) for node in self.graph.nodes}
        edge_labels = {(u, v): self.graph[u][v]["weight"] for u, v in self.graph.edges}

        plt.figure(figsize=(12, 8))

        # Draw graph
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_color="lightblue", edge_color="gray",
                node_size=2000, font_size=12, font_weight="bold")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)

        # Create a structured legend mapping binary positions to constraints
        if self.constraint_order:
            legend_text = "\n".join([f"{i}: {constraint}" for i, constraint in enumerate(self.constraint_order)])

            # Use matplotlib text box for cleaner display
            plt.gcf().text(0.85, 0.5, f"Legend:\n\n{legend_text}", fontsize=10, verticalalignment='center',
                        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

        plt.title("State Transition Graph (Binary Encoding)")

        # Save
        if save_path:
            plt.savefig(save_path, dpi=300)
            # print(f"Graph saved at: {save_path}")
            plt.close()  # Close after saving

    def prune_graph(self):
        """Remove redundant edges where direct transitions exist."""
        edges_to_remove = []
        for node in self.graph:
            successors = list(self.graph.successors(node))
            for i, s1 in enumerate(successors):
                for s2 in successors[i+1:]:
                    if nx.has_path(self.graph, s1, s2):
                        edges_to_remove.append((node, s2))
        self.graph.remove_edges_from(edges_to_remove)

    def compute_reward(self):
        """Assign rewards based on transition frequency."""
        if hasattr(self, "cached_rewards"):
            return self.cached_rewards  # Return cached rewards if available

        reward_function = {}
        for u, v, data in self.graph.edges(data=True):
            frequency = data["weight"]
            reward_function[(u, v)] = 1 / (frequency + 1e-5)  # Inverse frequency

        self.cached_rewards = reward_function  # Cache the result
        return reward_function