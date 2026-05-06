import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


"""
BioAnomaly : python package for anomaly detection in biological data.
Author : Ihsane ERRAMI
Date : 06/2024
"""
class BioAnomaly:
    def __init__(self, data_path, labels_path = None):
        self.data_path = data_path
        self.labels_path= labels_path
        self.data = None
        self.labels = None
        self.X_scaled = None
        self.X_pca = None
        self.X_tsne = None  
        self.iso_scores = None
        self.iso_preds = None
        self.recon_errors = None
        self.ae_preds = None
        self.combined_score = None
        self.model = None

    # Load data from CSV file
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path, index_col = 0, sep=",")
            print(f"Data loaded successfully.\n{self.data.shape[0]} samples and {self.data.shape[1]} genes.")
            # load labels if provided
            if self.labels_path :
                self.labels = pd.read_csv(self.labels_path, index_col = 0, sep=",")
                print(f"Labels loaded successfully.\n{self.labels['Class'].value_counts().to_dict()}")
        except Exception as e:
            print(f"Error loading data: {e}")

    #preprocessing data handling missing values
    def preprocess_data(self, n_components = 0.95):
        if self.data is not None:
            self.data.fillna(self.data.mean(), inplace=True)

            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.data)

            #PCA
            pca = PCA(n_components = n_components, random_state= 42)
            self.X_pca = pca.fit_transform(self.X_scaled)
            print(f"PCA completed. {self.data.shape[1]} genes reduced to {self.X_pca.shape[1]} components.")
            
            print("Data preprocessed successfully.")
        else:
            print("Data not loaded. Please load the data first.")

    """ 
    isolation forest 
    isolate anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
    """
    def isolation_forest(self, contamination = 0.05, n_estimator = 200):
        iso = IsolationForest(contamination = contamination, n_estimators = n_estimator, random_state= 42, n_jobs = -1)

        # -1 if anomalies, 1 if normal
        self.iso_preds = iso.fit_predict(self.X_pca)
        #lower = more anomalous 
        self.iso_scores = iso.decision_function(self.X_pca)
        n_anom = (self.iso_preds == -1).sum()
        print(f"Isolation Forest completed. Detected {n_anom} anomalies.")

    """
    autoencoder : NN used to learn a compressed representation of the data. It consists of an encoder that maps the input to a lower-dimensional latent space and a decoder that reconstructs the input from the latent representation. The model is trained to minimize the reconstruction error, which can be used as an anomaly score (higher error = more anomalous).
    """
    def _build_autoencoder(self, input_dim, latent_dim=32):
        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256), nn.ReLU(),
                    nn.Linear(256, 128),       nn.ReLU(),
                    nn.Linear(128, 64),        nn.ReLU(),
                    nn.Linear(64, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),  nn.ReLU(),
                    nn.Linear(64, 128),         nn.ReLU(),
                    nn.Linear(128, 256),        nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))
        return AE()
    
    """
    auto_encoder training and anomaly detection. The reconstruction error is calculated as the mean squared error between the input and the output of the autoencoder. A threshold is set based on the distribution of reconstruction errors, and samples with errors above this threshold are classified as anomalies.
    """
    def run_autoencoder(self, latent_dim= 32, epochs = 100, batch_size=32, lr= 1e-3, threshold_pct = 95):
        scaler2 = MinMaxScaler()
        X_norm = scaler2.fit_transform(self.X_pca).astype(np.float32)
        X_tensor = torch.tensor(X_norm)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True, drop_last=True)

        self.model = self._build_autoencoder(X_norm.shape[1], latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total = sum(
                (lambda loss: (optimizer.zero_grad(), loss.backward(), optimizer.step(), loss.item())[-1])(
                    criterion(self.model(batch), batch)
                )
                for (batch,) in loader
            )
            if (epoch + 1) % 25 == 0:
                print(f"   Epoch {epoch+1}/{epochs}  loss={total/len(loader):.5f}")

        self.model.eval()
        with torch.no_grad():
            X_recon = self.model(X_tensor).numpy()

        self.recon_errors = np.mean((X_norm - X_recon) ** 2, axis=1)
        threshold = np.percentile(self.recon_errors, threshold_pct)
        self.ae_preds = (self.recon_errors > threshold).astype(int)
        n_anom = self.ae_preds.sum()
        print(f"Autoencoder: {n_anom} anomalies ({n_anom/len(self.ae_preds)*100:.1f}%) | threshold={threshold:.6f}")


if __name__ == "__main__":
    bio_anomaly = BioAnomaly("data.csv", "labels.csv")
    bio_anomaly.load_data()
    bio_anomaly.preprocess_data()
    bio_anomaly.isolation_forest() #isolation forest
    bio_anomaly.run_autoencoder() #autoencoder