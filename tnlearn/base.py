"""
Program name: BaseModel Class Usability
Purpose description: This script demonstrates the utility and fundamental operations of
                     the BaseModel class within the 'tnlearn' package. The BaseModel
                     serves as a parent class providing common functionality for machine
                     learning models such as model saving/loading, progress plotting during
                     training, and calculation of performance metrics.
                     It is used here as a base from which specialized models like MLPRegressor
                     and MLPClassifier are derived.
Last revision date: February 20, 2024
Known Issues: None identified at the time of the last revision.
Note: This overview assumes that the Visualization class and all dependencies of `tnlearn`
      are properly installed and functional. Torch is utilized for model operations, while
      performance metrics are computed using the sklearn library.
"""

import os
import torch
from tnlearn.utils import Visualization
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


class BaseModel:
    def __init__(self):
        # Initialization method of the BaseModel class that sets up a visualization tool
        self.visualization = Visualization()

    # Method to update the progress plot during training
    def plot_progress(self, loss, savefig=False, accuracy=None):
        # Update visualization with the current epoch, loss, and optional accuracy
        self.visualization.update(self.current_epoch, loss, accuracy, savefig=savefig)

    def save(self, path, filename):
        """Save the current model to the specified path with the given filename."""
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, filename)
        # Save the model's weights to the constructed path
        torch.save(self.net.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, path, filename, input_dim, output_dim):
        """Load a model from the specified path with the given filename."""
        full_path = os.path.join(path, filename)
        # Check if the specified model file exists
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"No model found at {full_path}")

        # Reconstruct the model's architecture before loading the weights
        self.build_model(input_dim=input_dim, output_dim=output_dim)
        self.net.load_state_dict(torch.load(full_path))
        self.net.eval()  # Set the model to evaluation mode after loading weights
        print(f"Model loaded from {full_path}")

    # The following methods calculate different evaluation metrics
    def calculate_auc(self, y_true, y_pred):
        # Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        return roc_auc_score(y_true, y_pred)

    def calculate_f1_score(self, y_true, y_pred):
        # Calculate the F1 score, a weighted average of precision and recall
        return f1_score(y_true, y_pred)

    def calculate_recall(self, y_true, y_pred):
        # Calculate the recall, the ability of the classifier to find all the positive samples
        return recall_score(y_true, y_pred)

    def calculate_precision(self, y_true, y_pred):
        # Calculate the precision, the ability of the classifier not to label a sample
        # as positive if it is negative
        return precision_score(y_true, y_pred)
