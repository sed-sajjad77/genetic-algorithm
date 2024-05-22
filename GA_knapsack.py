import numpy as np
from numpy import random as rn
import random as rd

# read file


def read_file(path):
    f = open(path, "r")
    knapsack = int(f.readline())
    number = int(f.readline())
    weight = f.readline().split()
    value = f.readline().split()
    weight = [int(numeric_string) for numeric_string in weight]
    value = [int(numeric_string) for numeric_string in value]
    return knapsack, number, weight, value


def initial_population(num_population, number):
    return (rn.randint(2, size=(num_population, number)))


def Fitness(population, knapsack_w, weight, value, num_population):
    fitness = np.empty(num_population)
    for i in range(num_population):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= knapsack_w:
            fitness[i] = S1
        else:
            fitness[i] = 0
    return fitness.astype(int)


def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i = 0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        offsprings[i, 0:crossover_point] = parents[parent1_index,
                                                   0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index,
                                                  crossover_point:]
        i = +1
    return offsprings


def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        if random_value > mutation_rate:
            continue
        int_random_value = rn.randint(0, offsprings.shape[1]-1)
        if mutants[i, int_random_value] == 0:
            mutants[i, int_random_value] = 1
        else:
            mutants[i, int_random_value] = 0
    return mutants


# main
path = "KnapsackData\Knapsack_05.txt"
knapsack_w, number, weight, value = read_file(path)
# print(knapsack_w, number, weight, value)
num_population = 50
num_parents = int(num_population/2)
num_offsprings = num_population - num_parents
population = initial_population(num_population, number)
max_fitness_all = 0
i = 0
while i < 1000:
    # print(population)
    fitness = Fitness(population, knapsack_w, weight, value, num_population)
    # print(fitness)
    parents = selection(fitness, num_parents, population)
    # print(parents)
    offsprings = crossover(parents, num_offsprings)
    mutants = mutation(offsprings)
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = mutants
    # print(len(population))
    fitness_last_gen = Fitness(population, knapsack_w,
                               weight, value, num_population)
    max_fitness_idx = np.where(fitness_last_gen == np.max(fitness_last_gen))
    max_fitness = np.max(fitness_last_gen)
    parameters = population[max_fitness_idx[0][0], :]
    if max_fitness > max_fitness_all:
        max_fitness_all = max_fitness
        the_best = np.array(parameters, copy=True)
        i = 0
    else:
        i += 1

print(max_fitness)
print(the_best)
