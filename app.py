from flask import Flask, render_template, request
import numpy as np
from src.hmm import HiddenMarkovModel
from src.baum_welch import baum_welch
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    graph_path = None

    if request.method == "POST":
        hidden_states = int(request.form["hidden_states"])
        max_iter = int(request.form["max_iter"])
        observations = list(map(int, request.form["observations"].split(",")))

        # Random initialization
        A = np.random.dirichlet(np.ones(hidden_states), size=hidden_states)
        B = np.random.dirichlet(np.ones(2), size=hidden_states)
        pi = np.random.dirichlet(np.ones(hidden_states))

        model = HiddenMarkovModel(A, B, pi)
        model, likelihoods = baum_welch(model, observations, max_iter=max_iter)

        result = {
            "A": model.A,
            "B": model.B,
            "pi": model.pi
        }

        # Plot likelihood graph
        plt.figure()
        plt.plot(likelihoods)
        plt.xlabel("Iteration")
        plt.ylabel("P(O | λ)")
        plt.title("Likelihood over Iterations")
        graph_path = "static/graph.png"
        plt.savefig(graph_path)
        plt.close()

    return render_template("index.html", result=result, graph_path=graph_path)

if __name__ == "__main__":
    app.run(debug=True)