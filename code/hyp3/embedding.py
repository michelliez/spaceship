import torch
import torch.nn as nn
import torch.optim as optim

class Embedder(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
        
# embed_cols = ['Destination', 'HomePlanet', 'CabinDeck', 'CabinSide', 'SocioEconStatus']
# numerical_cols = ['Age', 'CabinNumber', 'GroupSize', 'TotalSpending', 'NoSpend', 'LuxurySpend', 'EssentialSpend', 'CryoNoSpend', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP']s