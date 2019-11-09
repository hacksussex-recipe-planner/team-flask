import json

class GeneticAlgorithm(DataLoader_Mixin):
    """ Genetic Algorithm class"""
    
    def __init__(self):
        super().__init__()
    
    def run_algorithm(self):
        pass

    def evaluation(self, instance):
        pass

class DataLoader_Mixin:
    """ DataLoader_Mixin"""
    
    def __init__(self):
        pass

    def load_data(self, file_path: str):
        pass

    def json_load(self, json_file):
        pass

    def json_results_get(self):
        pass