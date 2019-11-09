import json
import random

from deap import base, creator, tools


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

        self.calories = dict_temp["calories"]
        self.proteins = dict_temp["proteins"]
        self.carbohydrates = dict_temp["carbohydrates"]
        self.fat = dict_temp["fat"]
        self.meals_per_day = dict_temp["meals_per_day"]
    
    def data_sample(self):
        with open(self.file_path) as json_read:
            dict_temp = json.load(json_read)
            instance = []
            for i in range(self.meals_per_day):
                key = random.sample(dict_temp.keys(), 1)[0]
                print(key)
                instance.append([
                    key, int(dict_temp[key]["calories"]),
                    int(dict_temp[key]["proteins"]), int(dict_temp[key]["carbohydrates"]),
                    int(dict_temp[key]["fat"])
                ])
            return instance

class GeneticAlgorithm(DataLoader_Mixin):
    """ Genetic Algorithm class"""
    def __init__(self, ind_size: int = 10):
        super().__init__()
        self.ind_size = ind_size
        # 0, because ID needs to be stored
        creator.create("FitnessMulti", base.Fitness, weights=(0, -1, -0.25, -0.25, -0.25, 
                                                              0, -1, -0.25, -0.25, -0.25, 
                                                              0, -1, -0.25, -0.25, -0.25))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        # Register functions
        self.toolbox.register("attribute", super().data_sample)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                              self.toolbox.attribute, n=self.ind_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("evaluate", self._evaluate)

    def run_algorithm(self, file_path: str, json_file) -> dict():
        super().data_load(file_path)
        super().model_input_load(json_file)
        keys = random.sample(list(self.data), self.meals_per_day)
        return [self.data[k] for k in keys]
        
    def _mutate(self, instance, mu, sigma, indpb):
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

        return (cals, prots, carbs, fats)

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    data_gen = DataLoader_Mixin()
    json_file = data_gen.data_load("./sample.json", True)
    print(ga.run_algorithm("./data.json", json_file))
    for i in range(3):
        instance = ga.data_sample()
        print(f"Evaluation metric: {ga._evaluate(instance)}\n Run on: {instance}")