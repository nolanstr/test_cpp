import numpy as np
import pickle

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function \
        import BayesFitnessFunction

STACK_SIZE = 36
POP_SIZE = 100

def make_population(dim):

    data = np.load(f"../data/d{dim}_data.npy")
    X = data[:,:-1]
    y = data[:,-1]
    training_data = ExplicitTrainingData(X, y)

    component_generator = ComponentGenerator(training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    #component_generator.add_operator("exp")
    #component_generator.add_operator("log")
    #component_generator.add_operator("pow")
    #component_generator.add_operator("sqrt")
    #component_generator.add_operator("sin")
    #component_generator.add_operator("cos")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True
                                       )

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')

    smc_hyperparams = {'num_particles':10,
                       'mcmc_steps':10,
                       'ess_threshold':0.75}
    multisource_info = None 
    random_sample_info = 50

    bff = BayesFitnessFunction(local_opt_fitness,
                                   smc_hyperparams,
                                   multisource_info,
                                   random_sample_info,
                                   ensemble=4)

    pop = []
    
    while len(pop)<POP_SIZE:
        model = agraph_generator()
        f = bff(model)
        if not np.isnan(f):
            pop.append(str(model))
            print(len(pop))

    f = open(f"d{dim}_population.pkl", "wb")
    pickle.dump(pop, f)
    f.close()

if __name__ == '__main__':
    for dim in [1,2,3]:
        make_population(dim)

