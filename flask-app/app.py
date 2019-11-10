from flask import Flask, request, jsonify
from ml.GeneticAlgorithm import GeneticAlgorithm

import os, sys

fileDir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
ga = GeneticAlgorithm()

@app.route('/')
def hello():
    return fileDir + r"\ml\data.json"

@app.route('/get_recipes', methods=['GET', 'POST'])    
def recipes_list():
    content = request.json["data"]
    data = []
    for i in range(len(content)):
        _, data_dict = ga.run_algorithm(fileDir + r"\ml\data.json", content[str(i)])
        data.append(data_dict)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)