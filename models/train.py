import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lstm_model import SentimentDataset, LSTMSentiment

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    best_model_state = None
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    try:
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for vectors, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                vectors = vectors.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                output = model(vectors.unsqueeze(1))
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for vectors, labels in val_loader:
                    vectors = vectors.to(device)
                    labels = labels.to(device)
                    output = model(vectors.unsqueeze(1))
                    loss = criterion(output, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(output, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            accuracy = correct / total
            val_accuracies.append(accuracy)
            
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Accuracy: {accuracy:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                print("New best model saved!")
            
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
                
    except Exception as e:
        print(f"Error during training: {e}")
        raise
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, {"train_losses": train_losses, 
                  "val_losses": val_losses, 
                  "val_accuracies": val_accuracies}

# Main execution
if __name__ == "__main__":
    # Load data
    vector_file_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\vectors3000.csv"
    label_file_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\label3000.csv"
    
    vectors = pd.read_csv(vector_file_path)
    labels = pd.read_csv(label_file_path)
    
    # Convert labels [0, 2, 4] to [0, 1, 2] and ensure integer type
    labels = labels.astype(int)  # First ensure integer type
    le = LabelEncoder()
    labels = le.fit_transform(labels.values.ravel())
    
    print(f"shape of vectors: {vectors.shape}, shape of labels: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(vectors, labels, test_size=0.2, random_state=42)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Create data loaders
    train_dataset = SentimentDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = SentimentDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = 300  # tf-idf dimension
    hidden_dim = 128
    output_dim = 3  # Three classes: negative, neutral, positive
    n_layers = 2
    
    model = LSTMSentiment(input_dim, hidden_dim, output_dim, n_layers)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=5)
    
    # Save the model
    torch.save(model.state_dict(), r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\models\saved_model\lstm_sentiment_model.pth")
    print("Model saved successfully.")