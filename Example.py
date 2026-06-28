import torch
data = torch.load("./Data/cells.pt", weights_only=False)
print("Image shape:", data['train_images'].shape)
print("Unique labels:", torch.unique(data['train_labels']))
print("Num classes:", len(torch.unique(data['train_labels'])))