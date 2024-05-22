from numpy import random as rn
import numpy as np


def initial_population(num_population):
    return (rn.randint(8, size=(num_population, 8)))


def Fitness(population):
    fitness = []
    for i in range(len(population)):
        collision = 0
        for j in range(len(population[i])):
            k = j+1
            while k < len(population[i]):
                if population[i][j] == population[i][k]:
                    collision += 1
                elif population[i][j] + j == population[i][k] + k:
                    collision += 1
                elif population[i][j] - j == population[i][k] - k:
                    collision += 1
                k += 1
        fitness.append(28 - collision)
    return (fitness)


def Relative_selection(fitness, population):
    relative = []
    list1 = list(range(0, len(population)))
    for i in range(len(population)):
        for j in range(fitness[i]):
            relative.append(list1[i])
    rn.shuffle(relative)
    population_2 = []
    for i in range(len(population)):
        population_2.append(population[relative[i]])
    population_2 = np.array(population_2, int)
    return (population_2)


def multi_point_crossover(parents):
    children = parents.copy()
    for i in range(len(children)):
        if i % 2 == 1:
            x = rn.randint(len(children[i]))
            y = rn.randint(x, len(children[i]))
            y += 1
            cut = list(children[i-1][x:y])
            children[i-1][x:y] = children[i][x:y]
            children[i][x:y] = cut
    return (children)


def mutation(parents_children):
    darsad = 4/100
    n = int(darsad * len(parents))
    for i in range(n):
        x = rn.randint(len(parents_children))
        y = rn.randint(len(parents_children[x]))
        z = rn.randint(8)
        parents_children[x][y] = z
    return (parents_children)


def steady_state_replacement(population, children, fitness):
    darsad = 10/100
    n = int(darsad * len(population))
    m = len(population) - n
    rn.shuffle(population)

    new_population = []
    for i in range(m):
        new_population.append(population[i])

    for i in range(n):
        j = fitness.index(max(fitness))
        new_population.append(children[j])
        fitness[j] = 0

    population = np.array(new_population, int)
    return (population)


def generational_replacement(children):
    return (children)


def elitism(population, children, fitness_children, fitness_population):
    new_population = []
    j = fitness_children.index(min(fitness_children))
    for i in range(len(children)):
        if i != j:
            new_population.append(children[i])
    j = fitness_population.index(max(fitness_population))
    new_population.append(population[j])
    population = np.array(new_population, int)
    rn.shuffle(population)
    return (population)


def selection_method(population, children, fitness_children, fitness_population):
    darsad = 10/100
    n = int(darsad * len(population))
    m = len(population) - n
    new_population = []
    for i in range(m):
        j = fitness_population.index(max(fitness_population))
        new_population.append(population[j])
        fitness[j] = 0
    for i in range(n):
        j = fitness_children.index(max(fitness_children))
        new_population.append(children[j])
        fitness[j] = 0
    population = np.array(new_population, int)
    rn.shuffle(population)
    return (population)


if __name__ == '__main__':
    num_population = 50
    population = initial_population(num_population)
    fitness = Fitness(population)
    active = True
    epock = 0
    while active:
        best = max(fitness)
        if best == 28:
            active = False
            i = fitness.index(best)
            print(population[i])
            print(epock)
        parents = Relative_selection(fitness, population)
        children = multi_point_crossover(parents)
        children = mutation(children)
        fitness_children = Fitness(children)
        population = steady_state_replacement(
            population, children, fitness_children)
        # population = generational_replacement(children)
        # population = elitism(population, children, fitness_children, fitness)
        # population = selection_method(
        #     population, children, fitness_children, fitness)
        fitness = Fitness(population)
        epock += 1
        if epock % 500 == 0:
            print(epock)
            print(best)
