from z3 import *

# Define the predicates and numerical fluents
facing_iron = Bool('facing_iron')
facing_nothing = Bool('facing_nothing')
holding_iron_sword = Bool('holding_iron_sword')
inventory_iron = Int('inventory_iron')
inventory_iron_sword = Int('inventory_iron_sword')

# Define the current state (this should be parsed from the problem file)
current_state = {
    'facing_iron': False,
    'facing_nothing': True,
    'holding_iron_sword': False,
    'inventory_iron': 0,
    'inventory_iron_sword': 0
}

# Define the constraints based on the target grounding "iron"
constraints = [
    facing_iron,                   # c1: facing(iron)
    Not(facing_iron),              # c2: not facing(iron)
    inventory_iron > 0,            # c3: inventory(iron) > 0
    inventory_iron == 0,           # c4: inventory(iron) == 0
    holding_iron_sword,            # c5: holding(iron_sword)
    Not(holding_iron_sword)        # c6: not holding(iron_sword)
]

# Initialize the Z3 solver
solver = Solver()

# Function to evaluate constraints and construct the augmented feature vector
def evaluate_constraints_and_augment(current_state, constraints):
    # Clear the solver
    solver.reset()
    
    # Add the current state as assertions
    solver.add(facing_iron == current_state['facing_iron'])
    solver.add(facing_nothing == current_state['facing_nothing'])
    solver.add(holding_iron_sword == current_state['holding_iron_sword'])
    solver.add(inventory_iron == current_state['inventory_iron'])
    solver.add(inventory_iron_sword == current_state['inventory_iron_sword'])

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

# Evaluate the constraints for the current state
z = evaluate_constraints_and_augment(current_state, constraints)

# Print the augmented feature vector
print("Augmented feature vector z:", z)

# Original feature vector (example)
X = [0.5, 0.2, 0.3]  # This should be your actual feature vector

# Combine the original feature vector with the augmented feature vector
hat_X = X + z
print("Augmented feature vector hat_X:", hat_X)