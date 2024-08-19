(define (domain treasure_hunt_2_0)

(:requirements :strips :fluents :typing)

(:types 
    object
    facable - object
    holdable - object
    breakable - object
    unbreakable - object

    chest treasure iron_ore iron tree_log wood crafting_table
    iron_sword wall nothing platinum_ore titanium_ore gold_ore 
    silver_ore platinum_sword silver_sword titanium_sword gold_sword 
    gold silver titanium platinum - object

    chest - breakable 
    iron-ore - breakable
    platinum-ore - breakable
    gold_ore - breakable
    silver_ore - breakable
    titanium_ore - breakable
    tree-log - breakable

    wood - holdable
    iron - holdable
    silver - holdable
    gold - holdable
    titanium - holdable
    platinum - holdable
    treasure - holdable
    iron_sword - holdable
    silver_sword - holdable
    gold_sword - holdable
    titanium_sword - holdable
    platinum_sword - holdable
    nothing - holdable

    crafting_table - unbreakable
    wall - unbreakable
    nothing - unbreakable

    chest - facable
    iron_ore - facable
    platinum_ore - facable
    gold_ore - facable
    silver_ore - facable
    tree_log - facable
    crafting_table - facable
    wall - facable
    nothing - facable
)

(:predicates 
    (facing ?v0 - facable)
    (holding ?v0 - holdable)
    (produces ?b - breakable ?r - holdable)
)

(:functions 
    (inventory ?v0 - holdable)
    (world ?v0 - object)
)

(:action approach
    :parameters (?current - facable ?target - facable)
    :precondition (and (facing ?current))
    :effect (and (not (facing ?current)) (facing ?target))
)

(:action break
    :parameters (?target - breakable ?resource - holdable)
    :precondition (and 
        (holding nothing)(facing ?target) (> (world ?target) 0) (produces ?target ?resource))
    :effect (and 
        (not (facing ?target))
        (decrease (world ?target) 1)
        (increase (inventory ?resource) 1)
        (facing nothing)
    )
)

(:action select
    :parameters (?current - holdable ?target - holdable)
    :precondition (and 
        (> (inventory ?target) 0)
        (holding ?current)
    )
    :effect (and 
        (holding ?target) 
        (not (holding ?current)))
)

(:action craft_sword
    :parameters ()
    :precondition (and
        (facing crafting_table)
        (holding nothing)
        (>= (inventory wood) 1)
        (>= (inventory iron) 1)
    )
    :effect (and
        (decrease (inventory wood) 1)
        (decrease (inventory iron) 1)
        (increase (inventory iron_sword) 1)
    )
)
(:action obtain_treasure
    :parameters ()
    :precondition (and
        (facing chest)
        (holding iron_sword)
    )
    :effect (and
        (increase (inventory treasure) 1)
        (decrease (world chest) 1)
        (holding nothing)
        (facing nothing)
    )
)

)