(define (problem gridworld-problem)
(:domain treasure_hunt)

(:objects
    chest iron_ore tree_log crafting_table wall1 nothing - facable
    wood iron treasure iron_sword nothing - holdable
    chest iron_ore tree_log - breakable
    crafting_table wall nothing - unbreakable
)

(:init
    (= (world chest) 1)
    (= (world iron_ore) 1)
    (= (world tree_log) 1)
    (= (world crafting_table) 1)
    (= (world wall) 1)
    (= (world nothing) 1)
    (facing nothing)
    (holding nothing)
    (= (inventory wood) 0)
    (= (inventory iron) 0)
    (= (inventory iron_sword) 0)
    (= (inventory treasure) 0)

    ; always true
    (produces tree_log wood)
    (produces iron_ore iron)
)

(:goal (and
    ; (> (inventory iron) 0)
    ; (> (inventory wood) 0)
    ; (> (inventory iron_sword) 0)
    (> (inventory treasure) 1)
    ; (holding iron_sword)
))

)