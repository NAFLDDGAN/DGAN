import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class FeatureAttentionNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_index):
        super(FeatureAttentionNet, self).__init__()
        self.edge_index = edge_index  # Feature graph (edges between features)
        self.gat = GATConv(in_dim, 1, heads=1, concat=False)

    def forward(self, F):
        F_t = F.transpose(0, 1)  # shape: [num_features, num_patients]
        W_f = self.gat(F_t, self.edge_index)  # shape: [num_features, 1]
        return W_f.squeeze()  # shape: [num_features]


class PatientAttentionNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_index):
        super(PatientAttentionNet, self).__init__()
        self.edge_index = edge_index  # Patient graph
        self.gat = GATConv(in_dim, 1, heads=1, concat=False)

    def forward(self, F):
        W_p = self.gat(F, self.edge_index)  # shape: [num_patients, 1]
        return W_p.squeeze()  # shape: [num_patients]


class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_f, edge_index):
        x = self.gat1(X_f, edge_index)
        x = torch.relu(x)
        x = self.classifier(x, edge_index)
        return x


class DGAN(nn.Module):
    def __init__(self, in_feat_dim, hidden_dim, num_classes,
                 feat_edge_index, patient_edge_index):
        super(DGAN, self).__init__()
        self.feature_attention = FeatureAttentionNet(in_feat_dim, hidden_dim, feat_edge_index)
        self.patient_attention = PatientAttentionNet(in_feat_dim, hidden_dim, patient_edge_index)
        self.classifier = GraphClassifier(in_feat_dim, hidden_dim, num_classes)

    def forward(self, F, patient_edge_index):
        W_f = self.feature_attention(F)          # shape: [num_features]
        W_p = self.patient_attention(F)          # shape: [num_patients]

        X_f = F * W_f.unsqueeze(0)               # Apply W_f across features
        X_f = X_f * W_p.unsqueeze(1)             # Apply W_p across patients

        out = self.classifier(X_f, patient_edge_index)
        return out
