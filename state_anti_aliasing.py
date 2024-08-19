import re
from pysat.formula import CNF
from pysat.solvers import Solver

def parse_pddl_domain(domain_file):
    with open(domain_file, 'r') as file:
        data = file.read()

    # Extract types and their objects
    types_section = re.search(r':types\s+([\s\S]*?)\n\s*(:|\()', data)
    types_dict = {}
    if types_section:
        types_section = types_section.group(1)
        for line in types_section.split('\n'):
            line = line.strip()
            if '-' in line:
                parts = line.split('-')
                objects = parts[0].strip().split()
                typ = parts[1].strip()
                if typ not in types_dict:
                    types_dict[typ] = []
                types_dict[typ].extend(objects)

    # Extract predicates
    predicates_section = re.search(r':predicates\s+([\s\S]*?)\n\s*(:|\()', data)
    predicates_dict = {}
    if predicates_section:
        predicates_section = predicates_section.group(1)
        for line in predicates_section.split('\n'):
            line = line.strip()
            if line:
                match = re.match(r'\((\w+)', line)
                if match:
                    pred_name = match.group(1)
                    if pred_name not in predicates_dict:
                        predicates_dict[pred_name] = []
                    param_matches = re.findall(r'\?v\d - (\w+)', line)
                    predicates_dict[pred_name].extend(param_matches)

    # Extract functions (for inventory)
    functions_section = re.search(r':functions\s+([\s\S]*?)\n\s*(:|\()', data)
    functions_dict = {}
    if functions_section:
        functions_section = functions_section.group(1)
        for line in functions_section.split('\n'):
            line = line.strip()
            if line:
                match = re.match(r'\((\w+)', line)
                if match:
                    func_name = match.group(1)
                    if func_name not in functions_dict:
                        functions_dict[func_name] = []
                    param_matches = re.findall(r'\?v\d - (\w+)', line)
                    functions_dict[func_name].extend(param_matches)

    return types_dict, predicates_dict, functions_dict

def identify_related_groundings(types_dict, grounding_name):
    """
    Identify related groundings from the types dictionary based on the grounding name.
    """
    related = {}
    for typ, objects in types_dict.items():
        for obj in objects:
            if grounding_name in obj:
                if typ not in related:
                    related[typ] = []
                related[typ].append(obj)
    return related

def encode_formula(predicates, functions, related_groundings, var_map):
    """
    Encode the logical formula for the hint into CNF.
    """
    cnf = CNF()
    var_counter = len(var_map) + 1

    def get_var(var):
        nonlocal var_counter
        if var not in var_map:
            var_map[var] = var_counter
            var_counter += 1
        return var_map[var]

    # Ensure all titanium-related predicates are considered in the CNF clauses
    for predicate, types in predicates.items():
        for typ in types:
            if typ in related_groundings:
                clause = []
                for obj in related_groundings[typ]:
                    clause.append(get_var(f"{predicate}({obj})"))
                cnf.append(clause)
                print(f"CNF Clause for {predicate}({obj}):", clause)

    for function, types in functions.items():
        for typ in types:
            if typ in related_groundings:
                clause = []
                for obj in related_groundings[typ]:
                    clause.append(get_var(f"{function}({obj})"))
                cnf.append(clause)
                print(f"CNF Clause for {function}({obj}):", clause)

    return cnf, var_map

def populate_var_map(state, var_map):
    """
    Ensure the var_map includes all variables from the current state.
    """
    var_counter = len(var_map) + 1

    def get_var(var):
        nonlocal var_counter
        if var not in var_map:
            var_map[var] = var_counter
            var_counter += 1
        return var_map[var]

    for pred, arg in state.items():
        get_var(f"{pred}({arg})")

    return var_map

def check_satisfiability(cnf, state, var_map):
    """
    Check satisfiability of the state against the CNF formula.
    """
    solver = Solver()
    solver.append_formula(cnf)
    assumptions = [var_map[f"{pred}({arg})"] for pred, arg in state.items()]
    print("Assumptions for SAT Solver:", assumptions)
    is_satisfiable = solver.solve(assumptions=assumptions)
    solver.delete()
    return is_satisfiable

def transform_state(state, related_groundings, predicates, functions, var_map):
    """
    Transform the state based on the hint formula.
    """
    cnf, var_map = encode_formula(predicates, functions, related_groundings, var_map)
    var_map = populate_var_map(state, var_map)
    print("CNF Formula:", cnf)
    is_satisfiable = check_satisfiability(cnf, state, var_map)
    print("Is Satisfiable:", is_satisfiable)
    priority_bit = 1 if is_satisfiable else 0
    return {**state, 'priority_bit': priority_bit}

def main():
    domain_file = 'treasure_hunt_domain_2_0.pddl'
    grounding_name = 'titanium'

    # Parse domain
    types, predicates, functions = parse_pddl_domain(domain_file)
    print("Types:", types)
    print("Predicates:", predicates)
    print("Functions:", functions)

    # Identify related groundings based on the grounding name
    related_groundings = identify_related_groundings(types, grounding_name)
    print(f"{grounding_name.capitalize()}-related Groundings:", related_groundings)

    # Example state
    current_state = {
        'holding': 'iron',
        'facing': 'crafting_table'
    }

    # Variable map for CNF encoding
    var_map = {}

    # Transform the state
    transformed_state = transform_state(current_state, related_groundings, predicates, functions, var_map)
    print("Transformed State:", transformed_state)

if __name__ == '__main__':
    main()