from deap import base
from deap import creator
from deap import tools

import numpy as np

from matplotlib import pyplot as plt

import copy
from collections import namedtuple

import random
import pandas as pd
import math

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#===========================================================================================

vocab = None #Vocabulary after vectorization
vectorizer = None
tfidf_matrix = None

num_tokens = None
num_bits = None

iters_per_experiment = 10

max_generations = 1000
max_no_improvement = 200

#===========================================================================================

def list_to_int(l):
    bin_str = "".join(map(str, l))
    return int(bin_str, 2)

#===========================================================================================

def create_valid_individual(ind_class, func):

    ind = ind_class()

    for _ in range(2):
        temp = tools.initRepeat(ind_class, func, 1)
        ind += temp
        
        if temp[0] == 0:
            ind += tools.initRepeat(ind_class, func, 10)
        else:
            ind += [0]

            temp = tools.initRepeat(ind_class, func, 2)
            ind += temp

            if list_to_int(temp) in [0,1,2]:
                ind += tools.initRepeat(ind_class, func, 7)
            else:
                ind += [0]

                temp = tools.initRepeat(ind_class, func, 2)
                ind += temp

                if list_to_int(temp) in [0,1,2]:
                    ind += tools.initRepeat(ind_class, func, 4)
                else:
                    ind += [0,0,0,0]


    return ind    

#===========================================================================================

def valid_gene(l):

    #Token count is num_tokens
    #Valid range of indexes is [0, num_tokens - 1]
    if list_to_int(l) >= num_tokens:
        return False #invalid gene
    
    return True

#===========================================================================================

def repair_individual(ind):

    if not valid_gene(ind[0:num_bits]):
        ind[0] = 0

    if not valid_gene(ind[num_bits:]):
        ind[num_bits] = 0

#===========================================================================================

def create_individual(ind_class, func, n):
    i = tools.initRepeat(ind_class, func, n)

    repair_individual(i)

    return i

#===========================================================================================

def fitness(individual):

    repair_individual(individual)
    phrase = decode(individual)
    similarity = cosine_similarity(vectorizer.transform(phrase), tfidf_matrix).flatten()

    return (sum(sorted(similarity, reverse=True)[:10]), )

#===========================================================================================

def decode(individual):
    w1 = list_to_int(individual[0:num_bits])
    w2 = list_to_int(individual[num_bits:])

    word1 = vocab[w1]
    word2 = vocab[w2]

    return [word1 + " αλεξανδρε ουδις " + word2]

#===========================================================================================

if __name__ == "__main__":

    #Dataset
    #===========================================================================================
    dataset = pd.read_csv("iphi2802.csv", delimiter="\t", converters={"text": lambda x: x.replace("[","").replace("]","")}, usecols=["text", "region_main_id"])
    dataset = dataset.loc[dataset["region_main_id"] == 1693].reset_index(drop=True)["text"]

    #TF-IDF Vectorization
    #===========================================================================================
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)

    vocab = vectorizer.get_feature_names_out()

    num_tokens = tfidf_matrix.shape[1] #1457 tokens
    num_bits = math.floor(math.log2(num_tokens)) + 1 #11 bits
    
    print(f"Number of tokens: {num_tokens}\nNumber of bits: {num_bits}") #11 bits 

    #Create required classes
    #===========================================================================================
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    #Toolbox initialization
    #===========================================================================================
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", create_valid_individual, creator.Individual, toolbox.attr_bool)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)

    #Operators
    #===========================================================================================
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    #EXPERIMENT START
    #===========================================================================================
    
    Experiment = namedtuple('Experiment', ['popsize', 'pc', 'pm'])

    solution_score = 0
    solution_text = ""

    experiments = [
        Experiment(20, 0.6, 0),
        Experiment(20, 0.6, 0.01),
        Experiment(20, 0.6, 0.1),
        Experiment(20, 0.9, 0.01),
        Experiment(20, 0.1, 0.01),

        Experiment(200, 0.6, 0),
        Experiment(200, 0.6, 0.01),
        Experiment(200, 0.6, 0.1),
        Experiment(200, 0.9, 0.01),
        Experiment(200, 0.1, 0.01)
    ]

    excel_data = pd.DataFrame()
    plt.figure(figsize=(12, 8))

    #for----------------------------------------------------------------------------------------------------------

    for exp_num, exp in enumerate(experiments):

        best_solutions = np.zeros(iters_per_experiment)
        num_generations = np.zeros(iters_per_experiment)

        #The max number of generations seen across all iterations of this experiment
        #All arrays need to be padded with their last solution to reach this size
        max_generations_seen = 0

        best_per_generation = [[] for _ in range(iters_per_experiment)]

        toolbox.register("mutate", tools.mutFlipBit, indpb=exp.pm)

        #for----------------------------------------------------------------------------------------------------------

        for i in range(0, iters_per_experiment):

            generations = 0
            no_improvement_cnt = 0

            #Create and evaluate population
            #===========================================================================================
            population = toolbox.population(n=exp.popsize)

            for ind in population:
                ind.fitness.values = toolbox.evaluate(ind)

            #while----------------------------------------------------------------------------------------------------------

            while (generations < max_generations) and (no_improvement_cnt < max_no_improvement):

                elite_count = 1

                #Find initial elites
                #===========================================================================================
                previous_best = copy.deepcopy(tools.selBest(population, 1)[0])

                origin_elites = [copy.deepcopy(ind) for ind in tools.selBest(population, elite_count)]

                #Select individuals for temporary population
                #===========================================================================================
                selected = toolbox.select(population, len(population) - elite_count)

                #Choose individuals for crossover
                #We want so select pairs of individuals,to make sure that crossover can happen
                #===========================================================================================

                offspring = []
                unaltered = []

                for ind in selected:
                    if random.uniform(0,1) < exp.pc:
                        offspring.append(copy.deepcopy(ind))
                    else:
                        unaltered.append(copy.deepcopy(ind))

                #Make sure that we have pairs
                if len(offspring) % 2 == 1:

                    if len(unaltered) == 0: #We need to select one more parent, but we don't have anyone in unaltered
                        pos = random.randint(0, len(selected) - 1)
                        offspring.append(copy.deepcopy(selected[pos]))

                    else: #Choose a random person from unaltered
                        pos = random.randint(0, len(unaltered) - 1)
                        offspring.append(unaltered.pop(pos))

                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

                #We deleted fitness values of the parents, because they were replaced in-place by the children
                #We need to calculate fitness for the new children
                #We do this before mutation, so that we keep the elite unalterered
                #===========================================================================================

                for ind in offspring:
                    ind.fitness.values = toolbox.evaluate(ind)

                new_elites = [copy.deepcopy(ind) for ind in tools.selBest(offspring + unaltered + origin_elites, 1)]
                min_elite_score = new_elites[0].fitness.values[0]

                #Check if there is no improvement
                if previous_best.fitness.values[0] >= new_elites[0].fitness.values[0]:
                    no_improvement_cnt += 1
                else:
                    no_improvement_cnt = 0

                #Mutation
                #Make sure to re-evalueate those that get mutated
                ##===========================================================================================

                for mutant in offspring + unaltered:
                    
                    original_mutant = copy.deepcopy(mutant)
                    toolbox.mutate(mutant)

                    if mutant != original_mutant:
                        mutant.fitness.values = toolbox.evaluate(mutant)

                #New population + Statistics
                #===========================================================================================
                population[:] = offspring + new_elites + unaltered

                best_per_generation[i].append(new_elites[0].fitness.values[0])

                scores = [ind.fitness.values[0] for ind in population]
                print(f"\n[E{exp_num+1:02} I{i+1:02}] Generation {generations}")
                print(f"Max: {max(scores)}")
                print(f"Min: {min(scores)}")
                print(f"Avg: {sum(scores)/len(scores)}")

                print(decode(new_elites[0]))

                #Keep track of best solution, seen across ALL EXPERIMENTS
                if new_elites[0].fitness.values[0] > solution_score:
                    solution_score = new_elites[0].fitness.values[0]
                    solution_text = decode(new_elites[0])

                generations += 1

            #while----------------------------------------------------------------------------------------------------------

            max_generations_seen = max(max_generations_seen, generations)

            num_generations[i] = generations
            best_solutions[i] = tools.selBest(population, 1)[0].fitness.values[0]
        
        #for----------------------------------------------------------------------------------------------------------

        print(f"\nExperiment #{exp_num+1:02} stats:\n====================================================================================")
        print(f"Number of iterations: {iters_per_experiment}")
        print(f"Best solution average: {best_solutions.mean()}")
        print(f"Average number of generations: {num_generations.mean()}")

        excel_data = excel_data._append({"popsize": exp.popsize, "pc": exp.pc, "pm": exp.pm, "best_avg": best_solutions.mean(), "avg_generations": num_generations.mean()}, ignore_index=True)

        #Pad all arrays
        for arr in best_per_generation:
            arr.extend([arr[-1]] * (max_generations_seen - len(arr)))

        graph_data = np.mean(np.array(best_per_generation), axis=0)

        plt.gca().set_aspect('auto')

        plt.plot(graph_data, label=f"Exp. #{exp_num+1}")
        plt.xlabel('Generation')
        plt.ylabel('Average score')

    #for----------------------------------------------------------------------------------------------------------

    print(f"\nSolution: \"{solution_text[0]}\" with a score of {solution_score}")

    plt.legend()
    plt.savefig(f"results.png")


    while True:
        try:
            excel_data.to_excel("experiments.xlsx", index_label="Experiment")
            break
        except PermissionError:
            print("CLOSE THE FILE!")
