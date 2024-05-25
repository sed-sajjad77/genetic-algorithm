# Eight Queens Problem Solver with Genetic Algorithm
This code contains a Python implementation to solve the classic Eight Queens problem using a genetic algorithm approach. The Eight Queens problem involves placing eight chess queens on an 8×8 chessboard so that no two queens threaten each other; thus, a solution requires that no two queens share the same row, column, or diagonal.
### How it Works
The genetic algorithm employed in this project generates a population of potential solutions (arrangements of queens on the chessboard) and evolves them over multiple generations. The fitness of each solution is evaluated based on the number of conflicts (queens threatening each other), and genetic operators such as selection, crossover, and mutation are applied to produce new generations of solutions until a satisfactory solution is found or a termination condition is met.

# Ackley Function Minimization with Genetic Algorithm
This code contains a Python implementation to minimize the Ackley function using a genetic algorithm approach. The Ackley function is a benchmark optimization problem frequently used to test optimization algorithms. It is known for its multimodal and highly non-linear nature.
### Ackley Function
https://en.wikipedia.org/wiki/Ackley_function#:~:text=In%20mathematical%20optimization%2C%20the%20Ackley,in%20his%201987%20PhD%20dissertation.
### How it Works
The genetic algorithm implemented in this project generates a population of potential solutions (sets of 𝑥 and 𝑦 coordinates) and evolves them over multiple generations. The fitness of each solution is evaluated based on the value of the Ackley function, and genetic operators such as selection, crossover, and mutation are applied to produce new generations of solutions until a satisfactory solution (close to the global minimum) is found or a termination condition is met.

# Knapsack Problem Solver with Genetic Algorithm
This repository contains a Python implementation to solve the classic Knapsack problem using a genetic algorithm approach. The Knapsack problem involves selecting a combination of items with maximum value while not exceeding a given weight constraint.
### How it Works
The genetic algorithm implemented in this project generates a population of potential solutions (combinations of items) and evolves them over multiple generations. Each solution is represented as a binary string, where each bit represents whether an item is included or not. Genetic operators such as selection, crossover, and mutation are applied to produce new generations of solutions until a satisfactory solution (maximized value within weight constraints) is found or a termination condition is met.

# Quadcopter Aerial Training with Genetic Algorithm
This code contains a Python implementation to train a programmable quadcopter for efficient aerial maneuvers in a confined environment with obstacles and enemy soldiers. The goal is to eliminate all enemy soldiers without hitting the borders or throwing grenades into empty houses.
### Problem Description
The quadcopter operates in a 12 by 12 grid environment, where each cell represents a house. The environment is divided into regions, with the outer border acting as a restricted area for the quadcopter. Additionally, up to 20 enemy soldiers are randomly distributed throughout the environment. The quadcopter has the following actions available at each step:

1. Move one step to the east
2. Move one step to the west
3. Move one step to the south
4. Move one step to the north
5. Throw a grenade down
6. Do not move
7. Move randomly

The quadcopter receives rewards and penalties for its actions:

- Negative 5 points if it hits the forbidden area (borders)
- Negative 1 point if it throws a grenade into an empty house
- Positive 10 points if it eliminates a soldier with a grenade

The goal is to maximize the total points earned by the quadcopter while eliminating all enemy soldiers without hitting borders or wasting grenades.

## Requirements
- Python 3.x
- numpy (for array manipulation)
- Matplotlib (To see the results of quadcopter aerial training )