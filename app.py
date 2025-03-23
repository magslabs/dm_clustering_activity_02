from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load dataset from UCI
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
COLUMN_NAMES = [
    'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]
df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
true_labels = df['Class'].values
X = df.drop('Class', axis=1)

# Handle missing values (none in this dataset, but good practice)
X.fillna(X.mean(), inplace=True)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Ensure plots directory exists
os.makedirs("static/plots", exist_ok=True)

# Temporary file to store clustered CSV
clustered_csv_path = "static/clustered_data.csv"

def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set2')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    filepath = f'static/plots/{title}.png'
    plt.savefig(filepath)
    plt.close()
    return filepath

def run_dbscan():
    model = DBSCAN(eps=2, min_samples=5)
    labels = model.fit_predict(X_scaled)
    return create_result("DBSCAN", labels)

def run_gmm():
    model = GaussianMixture(n_components=3, random_state=42)
    labels = model.fit_predict(X_scaled)
    return create_result("GMM", labels)

def create_result(name, labels):
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    rand = adjusted_rand_score(true_labels, labels)
    img_path = plot_clusters(X_pca, labels, name)
    return {'name': name, 'labels': labels, 'sil': sil, 'rand': rand, 'img': img_path}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def run():
    results = [
        run_dbscan(),
        run_gmm()
    ]
    best_algorithm = max(results, key=lambda x: x['sil'])
    return render_template("results.html", results=results, best=best_algorithm)

@app.route("/download")
def download():
    return send_file(clustered_csv_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
