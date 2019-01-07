import random

from deap import base
from deap import creator
from deap import tools
import operator
import copy
import ujson
import numpy

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 100)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return sum(individual) - len(individual),
    #totale = 0

    #for i in range(0, len(individual), 2):
    #    if individual[i] == individual[i + 1]:
    #        totale -= 1
    #return totale,

# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


def binaryDistance(individual1, individual2):
    distance = 0
    
    for i in range(len(individual1)):
        if individual1[i] != individual2[i]:
            distance += 1

    return distance

#Usato per calcolare la distanza genetica
#toolbox.register("geneticDistance",binaryDistance, None, None)

#----------
def main():
    random.seed()
    standardResults = dict()
    modifiedResults = dict()

    populationSizes = [100,200,300]
    crossoverRates = [0.7,0.75,0.8,0.85]
    mutationRates = [0.125,0.15,0.2,0.25]
    recombinationRates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    testSize = 200
    for populationSize in populationSizes:
        for crossoverRate in crossoverRates:
            for mutationRate in mutationRates:
                standard = list()
                for i in range(testSize):
                    standard.append(runAlgorithm(populationSize,crossoverRate,mutationRate,0.1))

                standardResults[(populationSize, crossoverRate, mutationRate, 0)] = numpy.median(standard)
                
                for recombinationRate in recombinationRates:
                    modified = list()
                    for i in range(testSize):
                        modified.append(runAlgorithm(populationSize,crossoverRate,mutationRate,recombinationRate))
                    modifiedResults[(populationSize, crossoverRate, mutationRate, recombinationRate)] = numpy.median(modified)


    sortedStandard = sorted(standardResults.items(), key=operator.itemgetter(1))
    sortedModified = sorted(modifiedResults.items(), key=operator.itemgetter(1))

    print("====STANDARD====")
    for key, value in sortedStandard:
        print(" Population size: %s\n Crossover rate: %s\n Mutation rate: %s\n Evaluations: %s\n" % (key[0],key[1],key[2],value))
    print("====MODIFIED====")
    for key, value in sortedModified:
        print(" Population size: %s\n Crossover rate: %s\n Mutation rate: %s\n Recombination rate: %s\n Evaluations: %s\n" % (key[0],key[1],key[2],key[3],value)) 

    bestStandard = sortedStandard[0]
    print("====BEST STANDARD====")
    print(" Population size: %s\n Crossover rate: %s\n Mutation rate: %s\n Evaluations: %s\n" % (bestStandard[0][0],bestStandard[0][1],bestStandard[0][2],bestStandard[1]))

    bestModified = sortedModified[0]
    print("====BEST MODIFIED====")
    print(" Population size: %s\n Crossover rate: %s\n Mutation rate: %s\n Recombination rate: %s\n Evaluations: %s\n" % (bestModified[0][0],bestModified[0][1],bestModified[0][2], bestModified[0][3],bestModified[1]))
    
    #print("Standard: %s" % (sum(standardResults) / testSize,
    #sum(modifiedResults) / testSize))
def runAlgorithm(populationSize,crossoverRate, mutationRate, recombinationRate):

    evaluations = 0

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=populationSize)
    
    #print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    #print(" Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < 0 and evaluations < 100000:
        # A new generation
        g = g + 1
        #print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        parents = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list()

        for index in range(0,len(parents),2):
            parent1 = parents[index]
            parent2 = parents[index + 1]

            #child1 = toolbox.clone(parent1)
            #child2 = toolbox.clone(parent2)
            child1 = copy.copy(parent1)
            child1.fitness = copy.deepcopy(parent1.fitness)

            child2 = copy.copy(parent2)
            child2.fitness = copy.deepcopy(parent2.fitness)


            # Se non viene applicato il crossover i figli sono uguali ai
            # genitori
            if random.random() < crossoverRate:
                toolbox.mate(child1, child2)
                
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

            # mutate an individual with probability MUTPB
            if random.random() < mutationRate:
                toolbox.mutate(child1)
                del child1.fitness.values
            if random.random() < mutationRate:
                toolbox.mutate(child2)
                del child2.fitness.values
            

            #La prima generazione non ha nonni
            child1.parent1, child1.parent2 = parent1, parent2
            child2.parent1, child2.parent2 = parent1, parent2
            offspring.append(child1)
            offspring.append(child2)
            #Se il figlio è più simile ai genitori che ai nonni, allora tutto
            #ok

            firstChildFirstParent = binaryDistance(child1, parent1) <= binaryDistance(child1, parent1.parent1) and binaryDistance(child1, parent1) <= binaryDistance(child1, parent1.parent2)
            firstChildSecondParent = binaryDistance(child1, parent2) <= binaryDistance(child1, parent2.parent1) and binaryDistance(child1, parent2) <= binaryDistance(child1, parent2.parent2)

            secondChildFirstParent = binaryDistance(child2, parent1) <= binaryDistance(child2, parent1.parent1) and binaryDistance(child2, parent2) <= binaryDistance(child2, parent1.parent2)
            secondChildSecondParent = binaryDistance(child2, parent2) <= binaryDistance(child2, parent2.parent1) and binaryDistance(child2, parent2) <= binaryDistance(child2, parent2.parent2)

            if (firstChildFirstParent or firstChildSecondParent):
                child1.parent1, child1.parent2 = parent1, parent2
                newChildren.append(child1)

            #if ((secondChildFirstParent or secondChildSecondParent)) and
            #len(newChildren) < 2:
            #    child2.parent1, child2.parent2 = parent1, parent2
            #    nuoviFigli.append(child2)

            #if len(nuoviFigli) == 2:
            #    break

            #Quando trova una coppia compatibile imposta i genitori e
            #aggiungili alla nuova generazione
            offspring += nuoviFigli
            child2.parent1, child2.parent2 = parent1, parent2
            child1.parent1, child1.parent2 = parent1, parent2
            offspring.append(child1)
            offspring.append(child2)

    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

            evaluations += 1

            if ind.fitness.values < ind.parent1.fitness.values and ind.fitness.values < ind.parent2.fitness.values:
                if random.random() < 0.5:
                    recombinate(ind, ind.parent1, recombinationRate)
                else:
                    recombinate(ind, ind.parent2, recombinationRate)
            elif ind.fitness.values < ind.parent1.fitness.values:
                recombinate(ind, ind.parent1, recombinationRate)
            elif ind.fitness.values < ind.parent2.fitness.values:
                recombinate(ind, ind.parent2, recombinationRate)

        
        #print(" Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
    
    #print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print("Total evaluations: %s" % evaluations)

    return evaluations

def compare(individual1, individual2):
    risultato = list()
    for i in range(len(individual1)):
        if individual1[i] != individual2[i]:
            risultato.append(i)
    return risultato

def recombinate(individual1, individual2, recombinationRate):

    diff = compare(individual1, individual2)

    for i in range(int(recombinationRate * len(diff))):
        randomIndex = int(random.random() * len(diff))
        diffIndex = diff.pop(randomIndex)
        individual1[diffIndex] = 1 - individual1[diffIndex]

if __name__ == "__main__":
    main()

