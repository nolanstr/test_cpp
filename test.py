import numpy as np
import pickle
import time
import sys

from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function \
        import BayesFitnessFunction


def get_population(strings):
    from bingo.symbolic_regression.agraph.agraph import AGraph
    population = [AGraph(equation=string) for string in strings]
    cas = [i.command_array for i in population]

    try:
        from bingocpp import AGraph
        population = []
        for ca in cas:
            model = AGraph()
            model.command_array = ca
            str(model)
            population.append(model)
        return population
    except:
        return population

def time_code(dim, run, n=100):

    data = np.load(f"data/d{dim}_data.npy")
    X = data[:,:-1]
    y = data[:,-1]
    training_data = ExplicitTrainingData(X, y)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')

    smc_hyperparams = {'num_particles':10,
                       'mcmc_steps':10,
                       'ess_threshold':0.75}
    multisource_info = None
    random_sample_info = None 

    bff = BayesFitnessFunction(local_opt_fitness,
                                   smc_hyperparams,
                                   multisource_info,
                                   random_sample_info,
                                   ensemble=15)
    evaluator = Evaluation(bff, redundant=True)#, multiprocess=52)
    times = np.empty(n)
    eval_success = np.empty(n)

    for i in range(n):
        f = open(f"population/d{dim}_population.pkl", "rb")
        strings = pickle.load(f)
        f.close()
        population = get_population(strings)

        ts = time.time()
        evaluator(population)
        t = time.time() - ts
        
        times[i] = t
        nan_evals = np.isnan([ind.fitness for ind in population]).sum()
        eval_success[i] = (len(population)-nan_evals) / len(population)
    data = np.vstack((times, eval_success))
    np.save(f"{run}_d{dim}_stats", data)

if __name__ == '__main__':
    run = sys.argv[-1] 
    for i in [1,2,3]:
        time_code(i, run, n=10)
