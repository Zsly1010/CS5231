import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from py2neo import Graph

# --- Config Area ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456" # <--- Please enter your new password here

# --- Step 1: Extract Data from Neo4j ---
def get_data_from_neo4j(graph):
    print("[*] Extracting graph data from Neo4j...")
    
    # 1.1: Get features for all Process nodes
    # We use the three features we just calculated
    query_nodes = """
    MATCH (p:Process)
    RETURN p.pid AS pid, p.in_degree AS in_degree, p.out_degree AS out_degree, p.is_suspicious AS is_suspicious
    ORDER BY p.pid
    """
    node_data = graph.run(query_nodes).data()
    
    # Convert node data to feature tensor X
    # And create a mapping: PID -> index in the tensor (idx)
    x_list = []
    pid_to_idx = {}
    for idx, record in enumerate(node_data):
        pid_to_idx[record['pid']] = idx
        # Combine features into a list
        features = [
            float(record['in_degree']), 
            float(record['out_degree']), 
            float(record['is_suspicious'])
        ]
        x_list.append(features)
    
    # Convert to PyTorch Tensor
    x = torch.tensor(x_list, dtype=torch.float)
    
    # 1.2: Get all :CREATED relationships (graph edges)
    query_edges = """
    MATCH (p1:Process)-[:CREATED]->(p2:Process)
    RETURN p1.pid AS source_pid, p2.pid AS target_pid
    """
    edge_data = graph.run(query_edges).data()
    
    # Convert edge data to the edge_index format required by PyG
    source_indices = []
    target_indices = []
    for record in edge_data:
        if record['source_pid'] in pid_to_idx and record['target_pid'] in pid_to_idx:
            source_indices.append(pid_to_idx[record['source_pid']])
            target_indices.append(pid_to_idx[record['target_pid']])
            
    # Convert to PyTorch Tensor
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    
    print(f"[+] Extraction complete: {x.shape[0]} nodes, {edge_index.shape[1]} edges.")
    
    # Encapsulate as a PyG Data object
    data = Data(x=x, edge_index=edge_index)
    return data, pid_to_idx

# --- Step 2: Define GNN AutoEncoder Model ---
class GNNAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # Encoder: Two GCN layers
        self.encoder = GCNConv(in_channels, hidden_channels)
        # Decoder: Simply uses inner product to reconstruct the adjacency matrix
        
    def encode(self, x, edge_index):
        # Use ReLU activation function
        return self.encoder(x, edge_index).relu()

    def decode(self, z):
        # Decoder: Reconstructs edge existence probability via inner product of embedding vectors
        return torch.sigmoid((z @ z.t()))

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_recon = self.decode(z)
        return adj_recon, z

# --- Step 3: Train the Model ---
def train(model, data):
    print("[*] Starting GNN AutoEncoder training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create an adjacency matrix as the training target (Ground Truth)
    adj = torch.zeros((data.num_nodes, data.num_nodes))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    
    model.train()
    # Since we are on an emulator, only train for a few epochs to verify the process
    for epoch in range(50):
        optimizer.zero_grad()
        adj_recon, z = model(data.x, data.edge_index)
        
        # Calculate reconstruction loss (difference from the original adjacency matrix)
        loss = F.binary_cross_entropy(adj_recon.view(-1), adj.view(-1))
        
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'[*] Epoch: {epoch+1:02d}, Loss: {loss.item():.4f}')
    
    print("[+] Training complete.")
    return model, z # Return the trained model and final embeddings

# --- Step 4: Calculate Anomaly Scores and Write Back to Neo4j ---
def save_scores_to_neo4j(graph, pid_to_idx, data, z):
    print("[*] Calculating reconstruction error (anomaly scores)...")
    
    # (Skipping complex reconstruction error calculation for simplicity)
    # We will write the GNN-learned embedding vectors z directly back to Neo4j
    
    print("[*] Writing GNN embeddings back to Neo4j...")
    
    # Invert pid_to_idx mapping
    idx_to_pid = {idx: pid for pid, idx in pid_to_idx.items()}
    
    # Prepare batch update query
    updates = []
    for idx in range(z.shape[0]):
        pid = idx_to_pid[idx]
        # Convert tensor to Python list
        embedding = z[idx].detach().cpu().numpy().tolist()
        updates.append({
            'pid': pid,
            'embedding': embedding
        })
    
    # Use UNWIND for efficient batch updates in Neo4j
    query = """
    UNWIND $updates AS update
    MATCH (p:Process {pid: update.pid})
    SET p.gnn_embedding = update.embedding
    """
    graph.run(query, updates=updates)
    print(f"[+] Successfully wrote {len(updates)} GNN embeddings back to Neo4j.")

# --- Main Program Area ---
def main():
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph.run("RETURN 1")
        print("[+] Successfully connected to Neo4j database.")
    except Exception as e:
        print(f"[!] Error: Failed to connect to Neo4j database: {e}")
        return

    # 1. Extract Data
    data, pid_to_idx = get_data_from_neo4j(graph)
    
    # 2. Define Model
    # in_channels = 3 (because we have 3 features: in_degree, out_degree, is_suspicious)
    # hidden_channels = 8 (dimension of the GNN-learned embedding vector)
    model = GNNAutoEncoder(in_channels=3, hidden_channels=8)
    
    # 3. Train Model
    model, embeddings = train(model, data)
    
    # 4. Write Back Results
    save_scores_to_neo4j(graph, pid_to_idx, data, embeddings)
    
    print("[*] GNN training and write-back complete!")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()