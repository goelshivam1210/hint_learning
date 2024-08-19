import re
from z3 import *

# Function to manually parse the PDDL domain file
def parse_pddl_domain(domain_file):
    with open(domain_file, 'r') as file:
        domain_content = file.read()
    return domain_content

# Function to manually parse the PDDL problem file and extract the current state
def parse_pddl_problem(problem_file):
    current_state = {}

    with open(problem_file, 'r') as file:
        lines = file.readlines()
    
    in_init = False
    for line in lines:
        line = line.strip()
        if line.startswith("(:init"):
            in_init = True
        elif line.startswith("(:goal"):
            in_init = False
        
        if in_init:
            # Handle predicates
            pred_match = re.match(r'\(([^ ]+) ([^ ]+)\)', line)
            if pred_match:
                pred_name = pred_match.group(1)
                pred_terms = pred_match.group(2)
                state_key = f"{pred_name}_{pred_terms}"
                current_state[state_key] = True
            
            # Handle numerical fluents
            fluent_match = re.match(r'\(= \(([^ ]+) ([^ ]+)\) ([0-9]+)\)', line)
            if fluent_match:
                fluent_name = fluent_match.group(1)
                fluent_terms = fluent_match.group(2)
                fluent_value = int(fluent_match.group(3))
                state_key = f"{fluent_name}_{fluent_terms}"
                current_state[state_key] = fluent_value

    return current_state

# Function to define Z3 variables for predicates and numerical fluents
def define_z3_variables(current_state):
    z3_vars = {}
    for key in current_state:
        if isinstance(current_state[key], bool):
            z3_vars[key] = Bool(key)
        else:
            z3_vars[key] = Int(key)
    return z3_vars

# Function to evaluate constraints and construct the augmented feature vector
def evaluate_constraints_and_augment(current_state, z3_vars, constraints):
    # Initialize the Z3 solver
    solver = Solver()

    # Add the current state as assertions
    for key in current_state:
        if isinstance(current_state[key], bool):
            solver.add(z3_vars[key] == current_state[key])
        else:
            solver.add(z3_vars[key] == current_state[key])

    # Evaluate the constraints
    z = []
    for c in constraints:
        solver.push()
        solver.add(c)
        if solver.check() == sat:
            z.append(1)
        else:
            z.append(0)
        solver.pop()
    print("Constraints evaluated:", z)

    return z

# Define the domain and problem file paths
domain_file = 'treasure_hunt_domain_1_0.pddl'
problem_file = 'treasure_hunt_problem_1_0.pddl'

# Parse the PDDL domain file (not used further in this example)
domain_content = parse_pddl_domain(domain_file)

# Parse the PDDL problem file to get the current state
current_state = parse_pddl_problem(problem_file)

# Define Z3 variables based on the current state
z3_vars = define_z3_variables(current_state)

# Define the constraints based on the target grounding "iron"
constraints = [
    z3_vars.get('facing_iron', BoolVal(False)),                  # c1: facing(iron)
    Not(z3_vars.get('facing_iron', BoolVal(False))),             # c2: not facing(iron)
    z3_vars.get('inventory_iron', IntVal(0)) > 0,                # c3: inventory(iron) > 0
    z3_vars.get('inventory_iron', IntVal(0)) == 0,               # c4: inventory(iron) == 0
    z3_vars.get('holding_iron_sword', BoolVal(False)),           # c5: holding(iron_sword)
    Not(z3_vars.get('holding_iron_sword', BoolVal(False)))       # c6: not holding(iron_sword)
]

# Evaluate the constraints for the current state
z = evaluate_constraints_and_augment(current_state, z3_vars, constraints)

# Print the augmented feature vector
print("Augmented feature vector z:", z)

# Original feature vector (example)
X = [0.5, 0.2, 0.3]  # This should be your actual feature vector

# Combine the original feature vector with the augmented feature vector
hat_X = X + z
print("Augmented feature vector hat_X:", hat_X)