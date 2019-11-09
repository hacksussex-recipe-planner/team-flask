from flask import Flask, request, jsonify
from ml.GeneticAlgorithm import GeneticAlgorithm

app = Flask(__name__)
ga = GeneticAlgorithm()

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/get_recipes', methods=['GET', 'POST'])    
def recipes_list():
    content = request.json
    data = ga.run_algorithm(r"C:\Users\kacpe\Desktop\Github\team-flask\flask-app\ml\data.json", content)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)