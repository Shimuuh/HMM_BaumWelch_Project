from flask import Flask, render_template, request
import numpy as np
from src.hmm import HiddenMarkovModel
from src.baum_welch import baum_welch
import matplotlib.pyplot as plt
import os
import json

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    graph_path = None
    error = None

    if request.method == "POST":
        try:
            hidden_states = int(request.form["hidden_states"])
            max_iter = int(request.form["max_iter"])
            # Parse observations and determine M (number of unique observation symbols)
            observations = list(map(int, request.form["observations"].split(",")))
            M = len(set(observations))  # Number of unique observation symbols
            
            # Create model with N and M
            model = HiddenMarkovModel(hidden_states, M)
            
            # Train using Baum-Welch
            trained_model, likelihoods = baum_welch(model, observations, max_iter=max_iter)
            
            # Prepare results for display
            result = {
                "A": np.array2string(trained_model.A, precision=4, separator=', '),
                "B": np.array2string(trained_model.B, precision=4, separator=', '),
                "pi": np.array2string(trained_model.pi, precision=4, separator=', '),
                "likelihoods": likelihoods,
                "final_likelihood": likelihoods[-1] if likelihoods else 0,
                "iterations": len(likelihoods)
            }
            
            # Plot likelihood graph
            plt.figure(figsize=(10, 6))
            plt.plot(likelihoods, 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("P(O | λ)", fontsize=12)
            plt.title("Baum-Welch Convergence", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save graph
            graph_path = "static/graph.png"
            os.makedirs("static", exist_ok=True)
            plt.savefig(graph_path, dpi=100)
            plt.close()
            
        except Exception as e:
            error = str(e)
    
    return render_template("index.html", result=result, graph_path=graph_path, error=error)

if __name__ == "__main__":
    app.run(debug=True)