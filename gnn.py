"""
intial code to instantiate and train a graph neural network for eGFR prediction
"""
import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        return x
    
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # debug: print all types
        print(f"out type: {type(out)}, data.y type: {type(data.y)}")
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def GNN_dataloader(X, y, batch_size=32, shuffle=True):
    """
    Convert 1-D dataframes into PyTorch Geometric Data objects and return a DataLoader.
    """
    from torch_geometric.data import Data
    data_list = []
    for i in range(len(X)):
        x_tensor = torch.tensor(X.iloc[i].values, dtype=torch.float).unsqueeze(1)  # Node features
        edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)  # Dummy edge index
        y_tensor = torch.tensor([y.iloc[i]], dtype=torch.float)  # Target variable
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        data_list.append(data)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    
    fold_path = "/home/alecacciatore/ECML26/GNN4eGFR/"
    file_path = os.path.join(fold_path, "XY_temp.csv")
    out_path = os.path.join(fold_path, "features_importance_score", "xgb_scores")
    os.makedirs(out_path, exist_ok=True)

    # read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # the last column is the target variable and the first one is an index
    X = df.iloc[:, 1:-3] # TODO:: include general practitioner features?
    y_classif = df.iloc[:, -1]
    y_regress = df.iloc[:, -2]

    # print per-class distribution
    five_class_counts = y_classif.value_counts().sort_index()
    print("Class distribution:")
    print(y_classif.value_counts())

    # y_classif contains [I, II, IIIa, IIIb, IV, V] labels
    # convert them to numerical labels for classification
    label_mapping = {'I': 0, 'II': 1, 'IIIa': 1, 'IIIb': 1, 'IV': 1, 'V': 1}
    y_classif = y_classif.map(label_mapping)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_classif, test_size=0.2, random_state=42, stratify=y_classif)

    # convert DataFrame to PyTorch Geometric Data objects (dataloaders)
    train_loader = GNN_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    test_loader = GNN_dataloader(X_test, y_test, batch_size=32, shuffle=False)

    # instantiate the GNN model
    input_dim = 1  # since each feature is treated as a node with a single feature
    hidden_dim = 64
    output_dim = 2  # binary classification
    model = GNN(input_dim, hidden_dim, output_dim, num_layers=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')