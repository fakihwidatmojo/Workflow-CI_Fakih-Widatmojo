import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
import os
import dagshub

# Konfigurasi Dagshub
DAGSHUB_URI = "https://dagshub.com/fakihwidatmojo/Eksperimen_SML_Fakih-Widatmojo.mlflow" 

def load_data(path):
    return pd.read_csv(path)

def main():
    # 1. Load Data Processed
    print("Loading data...")
    try:
        data = load_data('online-sales-dataset_preprocessing.csv')
        # Drop CustomerID karena tidak dipakai untuk training, hanya ID
        X = data.drop(columns=['CustomerID'], errors='ignore')
    except FileNotFoundError:
        print("File online-sales-dataset_preprocessing.csv tidak ditemukan! Jalankan preprocessing dulu.")
        return

    # 2. Setup MLflow Tracking ke DagsHub
    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_experiment("Eksperimen K-Means Clustering")
    
    # 3. Hyperparameter Tuning Loop (Mencari K terbaik dari 2 sampai 6)
    print("Mulai Training & Tuning...")
    
    # List untuk menampung inertia (untuk Elbow Plot)
    inertia_list = []
    k_range = range(2, 7)

    # Mulai Parent Run
    with mlflow.start_run(run_name="K-Means Hyperparameter Tuning"):
        
        for k in k_range:
            with mlflow.start_run(run_name=f"K-Means-k{k}", nested=True):
                print(f"Training K-Means dengan k={k}...")
                
                # A. Model Training
                model = KMeans(n_clusters=k, init='k-means++', random_state=42)
                model.fit(X)
                
                # B. Evaluation Metrics
                score_silhouette = silhouette_score(X, model.labels_)
                inertia = model.inertia_
                inertia_list.append(inertia)
                
                # C. Manual Logging (Syarat Skilled/Advance)
                # 1. Log Parameters
                mlflow.log_param("n_clusters", k)
                mlflow.log_param("algorithm", "K-Means")
                
                # 2. Log Metrics
                mlflow.log_metric("silhouette_score", score_silhouette)
                mlflow.log_metric("inertia", inertia)
                
                # 3. Log Model
                mlflow.sklearn.log_model(model, f"model_k{k}")
                
                # D. Log Artifacts (Syarat Advance: Minimal 2 artefak tambahan)
                # Artefak 1: Scatter Plot Distribusi Cluster (Recency vs Frequency)
                data_clustered = data.copy()
                data_clustered['Cluster'] = model.labels_
                
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=data_clustered, x='Recency', y='Frequency', hue='Cluster', palette='viridis')
                plt.title(f'Cluster Distribution (K={k}) - Recency vs Frequency')
                plot_filename = f"cluster_plot_k{k}.png"
                plt.savefig(plot_filename)
                plt.close()
                
                mlflow.log_artifact(plot_filename) # Upload ke DagsHub
                os.remove(plot_filename) # Hapus file lokal setelah upload
                
                # Artefak 2: CSV hasil cluster
                csv_filename = f"result_k{k}.csv"
                data_clustered.to_csv(csv_filename, index=False)
                mlflow.log_artifact(csv_filename)
                os.remove(csv_filename)

        print("Tuning Selesai. Cek DagsHub untuk hasil.")

if __name__ == "__main__":
    dagshub.init(repo_owner='fakihwidatmojo', repo_name='Eksperimen_SML_Fakih-Widatmojo', mlflow=True)

    main()
