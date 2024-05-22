import numpy as np
from numpy import random as rn


def ackley(sol):
    x1, x2 = sol
    part_1 = -0.2*np.sqrt(0.5*(x1*x1 + x2*x2))
    part_2 = 0.5*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))
    value = np.exp(1) + 20 - 20*np.exp(part_1) - np.exp(part_2)
    return value


def initial_population(num_population, N, lower, upper):
    return rn.uniform(low=lower, high=upper, size=(num_population, N))


def Fitness(population):
    f = []
    fitness = np.zeros(len(population))
    for i in range(len(population)):
        f.append(round(ackley(population[i]), 3))
    f_sort = np.sort(f)
    i = len(population)
    for fit in f_sort:
        fitness[f.index(fit)] = i
        i -= 1
    return fitness


def Relative_selection(fitness, population):
    relative = []
    for i in range(len(population)):
        for j in range(round(fitness[i])):
            relative.append(population[i])
    rn.shuffle(relative)
    population_2 = []
    for i in range(len(population)):
        population_2.append(population[i])
    population_2 = np.array(population_2, float)
    return (population_2)


def crossover(parents):
    children = np.zeros((len(parents), 2))
    for i in range(len(parents)):
        if i % 2 == 1:
            children[i-1][0] = parents[i-1][0]
            children[i-1][1] = parents[i][1]
            children[i][0] = parents[i][0]
            children[i][1] = parents[i-1][1]
    child = []
    for i in range(len(children)):
        if i % 2 == 0:
            child.append(children[i])
    return (child)


def mutation(children, lower, upper):
    darsad = 4/100
    n = int(darsad * len(children))
    for i in range(n):
        child = rn.randint(len(children))
        mutate_idx = rn.randint(0, 2)
        random_value = rn.uniform(lower, upper)
        children[child][mutate_idx] = random_value
    return (children)


def steady_state_replacement(population, children, fitness):
    darsad = 10/100
    n = int(darsad * len(population))
    m = len(population) - n
    rn.shuffle(population)

    new_population = []
    for i in range(m):
        new_population.append(population[i])

    for i in range(n):
        j = np.where(fitness == max(fitness))
        new_population.append(children[j[0][0]])
        fitness[j] = 0

    population = np.array(new_population, float)
    return (population)


if __name__ == '__main__':
    N = 2
    num_population = 100
    lower = -5
    upper = 6
    population = initial_population(num_population, N, lower, upper)
    # print(population)
    fitness = Fitness(population)
    idx_best = np.where(fitness == max(fitness))
    best = population[idx_best[0][0]]
    epock = 0
    improvement = 0
    while improvement <= 2000:
        parents = Relative_selection(fitness, population)
        children = crossover(parents)
        children = mutation(children, lower, upper)
        fitness_children = Fitness(children)
        population = steady_state_replacement(
            population, children, fitness_children)
        fitness = Fitness(population)
        idx_best = np.where(fitness == max(fitness))
        best_epock = population[idx_best[0][0]]
        if (ackley(best_epock) < ackley(best)):
            improvement = 0
            best = best_epock
        epock += 1
        improvement += 1

        if epock % 500 == 0:
            print(f"epock: {epock}")
            print(f"x1={best[0]} and x2= {best[1]}")
            print(f"f(x)= {ackley(best)}")
            print("......")

    print(f"epock: {epock}")
    print(f"x1={best[0]} and x2= {best[1]}")
    print(f"f(x)= {ackley(best)}")
