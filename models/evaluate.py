import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict

from lstm_model import SentimentDataset, LSTMSentiment

class ModelEvaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained PyTorch model
            test_loader: DataLoader containing test data
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize containers for predictions and actual labels
        self.predictions = []
        self.true_labels = []
        self.loss = 0.0
        
    def evaluate(self) -> Dict:
        """
        Evaluate the model and compute all metrics.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for vectors, labels in self.test_loader:
                vectors = vectors.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(vectors.unsqueeze(1))
                loss = self.criterion(outputs, labels)
                self.loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions and true labels
                self.predictions.extend(predicted.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())
                total_samples += labels.size(0)
        
        # Calculate average loss
        self.loss /= len(self.test_loader)
        
        # Compute all metrics
        metrics = {
            'loss': self.loss,
            'accuracy': self._calculate_accuracy(),
            'classification_report': self._generate_classification_report(),
            'confusion_matrix': self._generate_confusion_matrix()
        }
        
        return metrics
    
    def _calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        return (np.array(self.predictions) == np.array(self.true_labels)).mean()
    
    def _generate_classification_report(self) -> Dict:
        """Generate detailed classification metrics."""
        # Map back to original labels [0, 2, 4]
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return classification_report(self.true_labels, self.predictions, 
                                  target_names=[label_map[i] for i in range(3)],
                                  output_dict=True)
    
    def _generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix."""
        return confusion_matrix(self.true_labels, self.predictions)
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot and optionally save confusion matrix visualization.
        
        Args:
            save_path: Path to save the confusion matrix plot
        """
        plt.figure(figsize=(10, 8))
        cm = self._generate_confusion_matrix()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_results(self, save_path: str):
        """
        Save evaluation results to a file.
        
        Args:
            save_path: Path to save the results
        """
        metrics = self.evaluate()
        
        # Convert classification report to DataFrame for better formatting
        clf_report = pd.DataFrame(metrics['classification_report']).transpose()
        
        # Create a detailed report
        with open(save_path, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=======================\n\n")
            
            f.write(f"Test Loss: {metrics['loss']:.4f}\n")
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write(clf_report.to_string())
            f.write("\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']))

# Example usage
if __name__ == "__main__":
    # Load your test data
    test_vector_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\vectors3000.csv"
    test_label_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\label3000.csv"
    
    # Load and preprocess test data
    test_vectors = pd.read_csv(test_vector_path)
    test_labels = pd.read_csv(test_label_path)
    
    # Convert labels using the same LabelEncoder used during training
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    test_labels = le.fit_transform(test_labels.values.ravel())
    
    # Create test dataset and loader
    test_dataset = SentimentDataset(test_vectors.values, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load your trained model
    model_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\models\saved_model\lstm_sentiment_model.pth"
    input_dim = 300
    hidden_dim = 128
    output_dim = 3
    n_layers = 2
    
    model = LSTMSentiment(input_dim, hidden_dim, output_dim, n_layers)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, test_loader)
    metrics = evaluator.evaluate()
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(save_path="confusion_matrix.png")
    
    # Save detailed results
    evaluator.save_results("evaluation_results.txt")