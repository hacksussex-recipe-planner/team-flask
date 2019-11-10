import json
import random

from deap import base, creator, tools
from deap.algorithms import eaSimple

class DataLoader_Mixin():
    """ DataLoader_Mixin"""
    def __init__(self):
        pass

    def data_load(self, file_path: str, return_data: bool = False):
        self.file_path = file_path
        with open(file_path) as json_read:
            if isinstance(json_read, type(dict())):
                return json_read
            if return_data == True:
                return json.load(json_read)
            else:
                self.data = json.load(json_read)

    def model_input_load(self, json_file):
        if isinstance(json_file, type(dict())):
            dict_temp = json_file
        else: 
            dict_temp = json.load(json_file)
        print(dict_temp)
        self.calories = dict_temp["calories"]
        self.proteins = dict_temp["proteins"]
        self.carbohydrates = dict_temp["carbohydrates"]
        self.fat = dict_temp["fat"]
        self.meals_per_day = dict_temp["meals_per_day"]
    
    def data_sample(self, ind):
        with open(self.file_path) as json_read:
            dict_temp = json.load(json_read)
            instance = []
            for i in range(self.meals_per_day):
                key = random.sample(dict_temp.keys(), 1)[0]
                instance.append([
                    key, int(dict_temp[key]["calories"]),
                    int(dict_temp[key]["proteins"]), int(dict_temp[key]["carbohydrates"]),
                    int(dict_temp[key]["fat"])
                ])
            return ind(instance)

    def data_sample_one(self):
        with open(self.file_path) as json_read:
            dict_temp = json.load(json_read)
            instance = []
            for i in range(self.meals_per_day):
                key = random.sample(dict_temp.keys(), 1)[0]
                instance.append([
                    key, int(dict_temp[key]["calories"]),
                    int(dict_temp[key]["proteins"]), int(dict_temp[key]["carbohydrates"]),
                    int(dict_temp[key]["fat"])
                ])
            return instance[0]

class GeneticAlgorithm(DataLoader_Mixin):
    """ Genetic Algorithm class"""
    def __init__(self, ind_size: int = 10):
        super().__init__()
        self.ind_size = ind_size
        # 0, because ID needs to be stored
        creator.create("FitnessMulti", base.Fitness, weights=(0, -5, -0.5, -0.5, -0.5))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        # Register functions
        self.toolbox.register("attribute", super().data_sample, ind=creator.Individual)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                              self.toolbox.attribute, n=self.ind_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxOnePoint) # no need to change
        self.toolbox.register("mutate", self._mutate, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate)

    def run_algorithm(self, file_path: str, json_file) -> dict():
        super().data_load(file_path)
        super().model_input_load(json_file)
        pop = self.toolbox.population(n=300)[0]      
        CXPB, MUTPB, NGEN = 0.5, 0.2, 20        
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit   
        print(f"fitnesses[0]: {fitnesses[0]}, fitnesses[1]: {fitnesses[1]}")
        print(f"pop[0]: {pop[0]} \n pop[1]: {pop[1]}\n pop[0]>pop[1]?: {pop[0]>pop[1]}")
        
        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            #print(f"offspring: {offspring}")
            #print(f"\npop: {pop}")
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values  

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring        
        return pop, self.convert_back_to_dict(pop)
    
    def convert_back_to_dict(self, result_arr):
        day = result_arr[0]
        ids = []
        result_dict = []
        for i in range(self.meals_per_day):
            id = day[i][0]
            result_dict.append(self.data[id])
        return result_dict

    def run_fake_algorithm(self, file_path: str, json_file) -> dict():
        super().data_load(file_path)
        super().model_input_load(json_file)
        keys = random.sample(list(self.data), self.meals_per_day)
        return [self.data[k] for k in keys]
        
    def _mutate(self, instance, indpb):
        indpb = [indpb]
        if len(indpb) == 1:
            indpb = list(indpb) * self.meals_per_day
        if len(indpb) != self.meals_per_day:
            print("Weird mutation error!")
            print(indpb)
            indpb = indpb[:1] * self.meals_per_day
            print(f"New probs: {indpb}")

        for i, proba in enumerate(indpb):
            rand_float = random.uniform(0, 1)
            if rand_float <= proba:
                instance_new = super().data_sample_one()
                instance[i] = instance_new
        return instance

    def _select(self, individuals, k, tournsize):
        pass

    def _evaluate(self, instance):
        cals = 0
        prots = 0
        carbs = 0
        fats = 0
        # Compute a squared error
        for i in range(self.meals_per_day):                
            cals += instance[i][1]
            prots += instance[i][2]
            carbs += instance[i][3]
            fats += instance[i][4]
        cals = (self.calories - cals) ** 2
        prots = (self.proteins - prots) ** 2
        carbs = (self.carbohydrates - carbs) ** 2
        fats = (self.fat - fats) ** 2
        return (0, cals, prots, carbs, fats)

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    data_gen = DataLoader_Mixin()
    json_file = data_gen.data_load("./sample.json", True)
    result, _ = ga.run_algorithm("./data.json", json_file)
    result_arr = []
    print(result)

    test = []

    for day in result:
        results_arr_temp = {"calories" : 0, "proteins" : 0, "carbs" : 0, "fat" : 0}
        for meal in day:
            results_arr_temp["calories"] += meal[1]
            results_arr_temp["proteins"] += meal[2]
            results_arr_temp["carbs"] += meal[3]
            results_arr_temp["fat"] += meal[4]
        test.append(results_arr_temp)
    print(test)
    print(f"result_array: {result_arr}")
    print(f"Desired intake: {[ga.calories, ga.proteins, ga.carbohydrates, ga.fat]}")