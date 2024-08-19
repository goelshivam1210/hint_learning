(define (problem gridworld-problem)
(:domain gridworld)

(:objects
    tree1 - tree
    iron_ore1 - iron_ore
    chest1 - chest
    crafting_table1 - crafting_table
    wood1 - wood
    iron1 - iron
    treasure1 - treasure
    iron_sword1 - iron_sword
    ; tree1 iron_ore1 chest1 crafting_table1 - breakable
    ; wood1 iron1 treasure1 iron_sword1 - holdable
)

(:init
    (= (world tree1) 1)
    (= (world iron_ore1) 1)
    (= (world chest1) 1)
    (= (world crafting_table1) 1)
    (facing chest1)
    (= (inventory wood1) 0)
    (= (inventory iron1) 0)
    (= (inventory iron_sword1) 0)
    (= (inventory treasure1) 0)
)

(:goal (and
    (holding treasure1)
))

)