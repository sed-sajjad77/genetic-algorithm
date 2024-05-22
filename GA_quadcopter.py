import numpy as np
import matplotlib.pyplot as plt


def create_environment():
    env = np.zeros((12, 12))
    for i in range(12):
        env[0, i] = 2
        env[i, 0] = 2
        env[11, i] = 2
        env[i, 11] = 2

    for _ in range(20):
        x, y = np.random.randint(1, 11, size=2)
        while env[x, y] == 1:
            x, y = np.random.randint(1, 11, size=2)
        env[x, y] = 1
    return env


# Step3
def perform_action(env, pos, gene):
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    state = 0
    for i, offset in enumerate(offsets):
        # print(i,offset)
        x, y = pos[0] + offset[0], pos[1] + offset[1]
        state += env[x, y] * (3 ** i)
    action = gene[int(state)]
    score = 0
    new_pos = list(pos)

    if action == 1:
        new_pos[1] += 1
    elif action == 2:
        new_pos[1] -= 1
    elif action == 3:
        new_pos[0] += 1
    elif action == 4:
        new_pos[0] -= 1
    elif action == 5:
        if env[pos[0], pos[1]] == 1:
            score += 10
            env[pos[0], pos[1]] = 0
        else:
            score -= 1
    elif action == 7:
        new_pos = [
            pos[0] + np.random.choice([-1, 1]), pos[1] + np.random.choice([-1, 1])]
    if env[new_pos[0], new_pos[1]] == 2:
        score -= 5
        new_pos = pos
    return score, new_pos


# Step 4
def simulate_drone(env, gene, start_pos=(1, 1)):
    score = 0
    position = list(start_pos)
    for _ in range(100):
        score_add, position = perform_action(env, position, gene)
        score += score_add
        # print(score)
        # position = new_position
    # s = input("...")
    # print(score)
    return score


def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    # print(len(child1))
    return child1, child2


def mutate(gene, rate=0.01):
    for index in range(len(gene)):
        if np.random.rand() < rate:
            gene[index] = np.random.randint(1, 8)
    return gene


def select_parents(population, scores, number):
    # print(fitness_scores/fitness_scores.sum())
    # if scores.sum() == 0:
    #     print(population)
    parents_indices = np.random.choice(
        np.arange(len(population)), size=number, replace=False, p=scores/scores.sum())
    return population[parents_indices]


def multi_point_crossover(parent1, parent2, points=3):
    crossover_points = np.sort(np.random.randint(1, len(parent1), size=points))
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(0, len(crossover_points), 2):
        if i+1 < len(crossover_points):
            child1[crossover_points[i]:crossover_points[i+1]
                   ] = parent2[crossover_points[i]:crossover_points[i+1]]
            child2[crossover_points[i]:crossover_points[i+1]
                   ] = parent1[crossover_points[i]:crossover_points[i+1]]
    return child1, child2

# step5


def genetic_algorithm(env, population, population_size=100, num_generations=100000, mutation_rate=0.05, elitism_size=5):
    best_score = -np.inf
    best_gene = None
    helpp = 0
    best_scores = []
    change_generations = []
    for generation in range(num_generations):
        fitness_scores = []
        for gene in population:
            # print(env)
            score = simulate_drone(env.copy(), gene)
            fitness_scores.append(score)
            # print(score)
        fitness_scores = np.array(fitness_scores)
        # print(fitness_scores)

        elite_indices = np.argsort(fitness_scores)[-elitism_size:]
        # print(fitness_scores)
        elite_individuals = population[elite_indices]
        # print(elite_individuals)

        # print(np.max(fitness_scores))
        if np.max(fitness_scores) > best_score:
            helpp = 0
            best_score = np.max(fitness_scores)
            best_gene = population[np.argmax(fitness_scores)].copy()
            change_generations.append(generation)
        best_scores.append(best_score)

        min_score = np.min(fitness_scores)
        if min_score <= 0:
            fitness_scores += abs(min_score) + 1

        # print(min(fitness_scores))

        # if generation%100 == 0:
        print(f"Generation {generation + 1}, Best Score: {best_score}")
        num_parents = int(population_size/2)

        parents = select_parents(population, fitness_scores, num_parents)
        # print(parents.shape)
        next_generation = []
        for i in range(0, len(parents), 2):
            # while len(next_generation) < population_size:
            # parent1, parent2 = select_parents(parents, fitness_scores, 2)
            parent1, parent2 = parents[i], parents[i+1]
            # child1, child2 = crossover(parent1, parent2)
            child1, child2 = multi_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])
        # population = np.array(next_generation)[:population_size]
        population = np.array(next_generation)

        population[:elitism_size] = elite_individuals
        # print(population.shape)
        if helpp == 500 or best_score == 200:
            return best_gene, best_score, change_generations, best_scores
        helpp += 1
    return best_gene, best_score, change_generations, best_scores


def main():
    # step1
    # create random env
    # env = create_environment()
    # print(env)

    # create my env for test
    env = np.array([[2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                    [2., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 2.],
                    [2., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
                    [2., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2.],
                    [2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
                    [2., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 2.],
                    [2., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 2.],
                    [2., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 2.],
                    [2., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 2.],
                    [2., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 2.],
                    [2., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
                    [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]])
    # print(env)

    # step2 and initial population
    gene_length = 3 ** 5
    population_size = 100
    population = np.random.randint(1, 8, (population_size, gene_length))
    # print(population.shape)

    best_gene, best_score, change_generations, best_scores = genetic_algorithm(
        env, population, population_size)
    print("Best Gene:", best_gene)
    print("Best Score:", best_score)

    plt.figure(figsize=(10, 6))
    plt.plot(best_scores, marker='o', linestyle='-',
             color='b', label='Best Score')

    for gen in change_generations:
        plt.scatter(gen, best_scores[gen], color='red', zorder=5,
                    label='Improvement' if gen == change_generations[0] else "")

    plt.title('Best Score per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Score')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
