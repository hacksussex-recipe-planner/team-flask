import json
import random

from deap import base, creator, tools

class DataLoader_Mixin():
    """ DataLoader_Mixin"""
    
    def __init__(self):
        pass

    def data_load(self, file_path: str, return_data: bool = False):
        with open(file_path) as json_read:
            if return_data == True:
                return json.load(json_read)
            else:
                self.data = json.load(json_read)

    def model_input_load(self, json_file):
        dict_temp = json_file
        
        self.calories = dict_temp["calories"]
        self.proteins = dict_temp["proteins"]
        self.carbohydrates = dict_temp["carbohydrates"]
        self.fat = dict_temp["fat"]
    
    def data_sample(n: int):
        pass

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
        self.toolbox.register("attribute", random.random) #change to sample data
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                              self.toolbox.attribute, n=self.ind_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("evaluate", self._evaluate)

    def run_algorithm(self, file_path: str, json_file) -> dict():
        super().data_load(file_path)
        super().model_input_load(json_file)
        
    def _evaluate(self, instance):
        cals = 0
        prots = 0
        carbs = 0
        fats = 0
        # Compute a squared error
        for i in range(super().meals_per_day):
            cals += (super().calories - instance[i+1]) ** 2
            prots += (super().proteins - instance[i+2]) ** 2
            carbs += (super().carbohydrates - instance[i+3]) ** 2
            fats += (super().fat - instance[i+4]) ** 2

        return (cals, prots, carbs, fats)

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    data_gen = DataLoader_Mixin()
    json_file = data_gen.data_load("./sample.json", True)
    ga.run_algorithm("./data.json", json_file)