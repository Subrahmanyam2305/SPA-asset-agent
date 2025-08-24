import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler
from datetime import datetime
from torch_geometric.utils import add_self_loops, to_undirected
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# --- Config
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
window_size = 20  # Reduced window size for more samples
correlation_threshold = 0.3  # Reduced threshold to capture more relationships
num_assets = len(tickers)

def load_data(start_date="2023-01-01", end_date="2025-01-01"):
    """Load and prepare stock price data"""
    print("Loading data...")
    prices = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = prices.pct_change().dropna()
    return prices, returns

def normalize_features(features):
    """Normalize features using RobustScaler for better handling of outliers"""
    scaler = RobustScaler()
    shape = features.shape
    flattened = features.reshape(-1, shape[-1])
    normalized = scaler.fit_transform(flattened)
    return normalized.reshape(shape)

def calculate_features(window_data):
    """Calculate all features for a window of data"""
    features_list = []
    
    for ticker in tickers:
        series = window_data[ticker]
        
        # Basic features
        basic_features = series.values
        
        # Technical indicators (using log returns for better numerical stability)
        # Handle negative values before log1p
        pct_change = series.pct_change()
        pct_change = pct_change.fillna(0)
        # Add 1 to make all values positive before log
        log_returns = np.log1p(pct_change + 1)
        
        # Use bfill() instead of fillna(method='bfill')
        sma5 = pd.Series(series).rolling(5).mean().bfill().values
        sma10 = pd.Series(series).rolling(10).mean().bfill().values
        std5 = pd.Series(log_returns).rolling(5).std().fillna(0).values
        std10 = pd.Series(log_returns).rolling(10).std().fillna(0).values
        mom5 = pd.Series(log_returns).rolling(5).sum().fillna(0).values
        mom10 = pd.Series(log_returns).rolling(10).sum().fillna(0).values
        
        # Combine all features for this asset
        asset_features = np.stack([
            basic_features,
            sma5,
            sma10,
            std5,
            std10,
            mom5,
            mom10
        ])
        
        features_list.append(asset_features)
    
    # Stack all assets' features
    features = np.stack(features_list)
    
    # Normalize features
    return normalize_features(features)

def build_dataset(returns_df, start_idx, end_idx):
    features, targets, edge_indices, edge_weights = [], [], [], []
    num_features = 7  # base + 6 technical indicators
    
    print(f"Building dataset from index {start_idx} to {end_idx}")
    
    for t in range(start_idx, end_idx):
        # Get window of returns
        window_returns = returns_df.iloc[t - window_size:t]
        
        # Calculate features
        feature_matrix = calculate_features(window_returns)  # Shape: [num_assets, num_features, window_size]
        
        # Reshape features to [num_assets, num_features * window_size]
        feature_vector = feature_matrix.reshape(num_assets, -1)
        features.append(torch.tensor(feature_vector, dtype=torch.float))
        
        # Target is next day's return direction
        next_returns = returns_df.iloc[t + 1]
        targets.append(torch.tensor((next_returns > 0).astype(int).values, dtype=torch.long))
        
        # Create edges based on correlations using log returns for better stability
        pct_change = window_returns.pct_change().fillna(0)
        # Add 1 to make all values positive before log
        log_returns = np.log1p(pct_change + 1)
        corr = log_returns.corr().values
        edges = []
        weights = []
        
        # Add edges based on correlation
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                if abs(corr[i, j]) > correlation_threshold:
                    edges.append([i, j])
                    weights.append(abs(corr[i, j]))
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Make the graph undirected
            edge_index = to_undirected(edge_index)
            # Add self-loops
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_assets)
            
            # Create edge weights including self-loops
            edge_weights_tensor = torch.ones(edge_index.size(1), dtype=torch.float)
            if len(weights) > 0:
                edge_weights_tensor[:len(weights)] = torch.tensor(weights, dtype=torch.float)
                edge_weights_tensor[len(weights):len(weights)*2] = torch.tensor(weights, dtype=torch.float)  # Undirected duplicates
        else:
            # If no edges, just create self-loops
            edge_index = torch.tensor([[i, i] for i in range(num_assets)], dtype=torch.long).t()
            edge_weights_tensor = torch.ones(num_assets, dtype=torch.float)
        
        edge_indices.append(edge_index)
        edge_weights.append(edge_weights_tensor)
    
    return features, targets, edge_indices, edge_weights

def prepare_datasets(returns, window_size):
    """Prepare training, validation, and test datasets"""
    print("Splitting data...")
    dates = returns.index
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    train_end_idx = int(len(dates) * train_ratio)
    val_end_idx = int(len(dates) * (train_ratio + val_ratio))

    print(f"Training period: {dates[window_size]} to {dates[train_end_idx]}")
    print(f"Validation period: {dates[train_end_idx+1]} to {dates[val_end_idx]}")
    print(f"Testing period: {dates[val_end_idx+1]} to {dates[-1]}")

    # Build datasets
    start_idx = window_size
    train_features, train_targets, train_edges, train_weights = build_dataset(
        returns, start_idx, train_end_idx
    )

    val_features, val_targets, val_edges, val_weights = build_dataset(
        returns, train_end_idx + 1, val_end_idx
    )

    test_features, test_targets, test_edges, test_weights = build_dataset(
        returns, val_end_idx + 1, len(returns) - 2
    )

    return (train_features, train_targets, train_edges, train_weights,
            val_features, val_targets, val_edges, val_weights,
            test_features, test_targets, test_edges, test_weights)

class StockGraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        # MLP layers
        x = self.dropout(x)
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

def evaluate(model, features, targets, edges, weights, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for t in range(len(features)):
            x_t = features[t].to(device)
            y_t = targets[t].to(device)
            edge_idx = edges[t].to(device)
            edge_wt = weights[t].to(device)
            
            out = model(x_t, edge_idx, edge_wt)
            loss = F.cross_entropy(out, y_t)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            correct += (pred == y_t).sum().item()
            total += y_t.size(0)
    
    return correct / total, total_loss / len(features)

def train_model(train_features, train_targets, train_edges, train_weights,
              val_features, val_targets, val_edges, val_weights,
              window_size, device=None):
    """Train the model and return the best model state"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate input size
    input_size = 7 * window_size  # num_features * window_size
    print(f"Input feature size: {input_size}")

    # Initialize model with smaller hidden size
    model = StockGraphModel(
        in_channels=input_size,
        hidden_channels=64,  # Reduced from 128
        out_channels=2
    ).to(device)

    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training loop
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    epochs = 200

    print("\n🚀 Training:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for t in range(len(train_features)):
            x_t = train_features[t].to(device)
            y_t = train_targets[t].to(device)
            edge_idx = train_edges[t].to(device)
            edge_wt = train_weights[t].to(device)

            optimizer.zero_grad()
            out = model(x_t, edge_idx, edge_wt)
            loss = F.cross_entropy(out, y_t)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {t}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y_t).sum().item()
            total += y_t.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_features)
        
        # Validation
        val_acc, val_loss = evaluate(model, val_features, val_targets, val_edges, val_weights, device)
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    return model, best_val_acc

def test_model(model, test_features, test_targets, test_edges, test_weights, device=None):
    """Test the model and return test metrics"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pt'))
    test_acc, test_loss = evaluate(model, test_features, test_targets, test_edges, test_weights, device)
    
    print(f"\n📊 Final Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return test_acc, test_loss

def get_model_recommendations(model, prices_df, confidence_threshold=0.6):
    """
    Get model predictions and format them into actionable recommendations.
    
    Args:
        model: Trained StockGraphModel
        prices_df: DataFrame with latest price data
        confidence_threshold: Minimum confidence level for strong recommendations
    
    Returns:
        dict: Structured recommendations with confidence scores and market context
    """
    model.eval()
    
    # Get the most recent window of data
    latest_returns = prices_df.pct_change().dropna().iloc[-window_size:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate features for the latest window
    feature_matrix = calculate_features(latest_returns)
    feature_vector = feature_matrix.reshape(num_assets, -1)
    x = torch.tensor(feature_vector, dtype=torch.float).to(device)
    
    # Create edges for the latest window
    pct_change = latest_returns.pct_change().fillna(0)
    # Add 1 to make all values positive before log
    log_returns = np.log1p(pct_change + 1)
    corr = log_returns.corr().values
    edges = []
    weights = []
    
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            if abs(corr[i, j]) > correlation_threshold:
                edges.append([i, j])
                weights.append(abs(corr[i, j]))
    
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_assets)
        edge_weights = torch.ones(edge_index.size(1), dtype=torch.float)
    else:
        edge_index = torch.tensor([[i, i] for i in range(num_assets)], dtype=torch.long).t()
        edge_weights = torch.ones(num_assets, dtype=torch.float)
    
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    
    # Get model predictions
    with torch.no_grad():
        logits = model(x, edge_index, edge_weights)
        probabilities = F.softmax(logits, dim=1)
    
    # Calculate price changes and volatility
    price_changes = prices_df.pct_change().iloc[-1] * 100
    volatility = prices_df.pct_change().rolling(window=20).std().iloc[-1] * 100
    
    # Prepare recommendations
    recommendations = {
        "timestamp": datetime.now().isoformat(),
        "model_confidence": {
            "average_confidence": float(probabilities.max(dim=1)[0].mean()),
            "prediction_spread": float(probabilities.std()),
        },
        "market_context": {
            "recent_volatility": volatility.to_dict(),
            "24h_price_change": price_changes.to_dict()
        },
        "predictions": {}
    }
    
    # Generate recommendations for each asset
    for i, ticker in enumerate(tickers):
        prob = probabilities[i]
        prediction = int(prob.argmax())
        confidence = float(prob.max())
        
        sentiment = "neutral"
        if confidence > confidence_threshold:
            sentiment = "bullish" if prediction == 1 else "bearish"
        
        recommendations["predictions"][ticker] = {
            "sentiment": sentiment,
            "confidence": confidence,
            "price_change_24h": float(price_changes[ticker]),
            "volatility": float(volatility[ticker]),
            "correlation_strength": float(abs(corr[i]).mean()),
        }
    
    return recommendations

if __name__ == "__main__":
    # Load data
    prices, returns = load_data()
    
    # Prepare datasets
    (train_features, train_targets, train_edges, train_weights,
     val_features, val_targets, val_edges, val_weights,
     test_features, test_targets, test_edges, test_weights) = prepare_datasets(returns, window_size)
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_val_acc = train_model(
        train_features, train_targets, train_edges, train_weights,
        val_features, val_targets, val_edges, val_weights,
        window_size, device
    )
    
    # Test model
    test_acc, test_loss = test_model(
        model, test_features, test_targets, test_edges, test_weights, device
    )
    
    # Generate recommendations
    print("\n📈 Generating Trading Recommendations...")
    recommendations = get_model_recommendations(model, prices)
    
    # Save recommendations to file
    with open('latest_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nRecommendations Summary:")
    print(f"Average Confidence: {recommendations['model_confidence']['average_confidence']:.2f}")
    print("\nAsset Sentiments:")
    for ticker, data in recommendations['predictions'].items():
        print(f"{ticker}: {data['sentiment'].upper()} (Confidence: {data['confidence']:.2f})")
    
    print("\nDetailed recommendations have been saved to 'latest_recommendations.json'")