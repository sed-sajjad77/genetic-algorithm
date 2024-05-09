# Eight Queens Problem Solver with Genetic Algorithm
This repository contains a Python implementation to solve the classic Eight Queens problem using a genetic algorithm approach. The Eight Queens problem involves placing eight chess queens on an 8×8 chessboard so that no two queens threaten each other; thus, a solution requires that no two queens share the same row, column, or diagonal.
### How it Works
The genetic algorithm employed in this project generates a population of potential solutions (arrangements of queens on the chessboard) and evolves them over multiple generations. The fitness of each solution is evaluated based on the number of conflicts (queens threatening each other), and genetic operators such as selection, crossover, and mutation are applied to produce new generations of solutions until a satisfactory solution is found or a termination condition is met.
#   Ackley Function Minimization with Genetic Algorithm
This repository contains a Python implementation to minimize the Ackley function using a genetic algorithm approach. The Ackley function is a benchmark optimization problem frequently used to test optimization algorithms. It is known for its multimodal and highly non-linear nature.
### Ackley Function
https://en.wikipedia.org/wiki/Ackley_function#:~:text=In%20mathematical%20optimization%2C%20the%20Ackley,in%20his%201987%20PhD%20dissertation.
### How it Works
The genetic algorithm implemented in this project generates a population of potential solutions (sets of 𝑥 and 𝑦 coordinates) and evolves them over multiple generations. The fitness of each solution is evaluated based on the value of the Ackley function, and genetic operators such as selection, crossover, and mutation are applied to produce new generations of solutions until a satisfactory solution (close to the global minimum) is found or a termination condition is met.
## Requirements
- Python 3.x
- numpy (for array manipulation)