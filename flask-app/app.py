from flask import Flask, request, jsonify
from ml.GeneticAlgorithm import GeneticAlgorithm

app = Flask(__name__)
ga = GeneticAlgorithm()

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/get_recipes/<uuid>', methods=['GET', 'POST'])    
def get_recipes(uuid):
    content = request.json
    data = ga.run_algorithm(r"C:\Users\kacpe\Desktop\Github\team-flask\ml\data.json", content)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)