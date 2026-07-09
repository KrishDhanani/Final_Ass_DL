"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1, fraction=0.1):
    d_path = Path(data_path) / f"{data}"
    print(d_path)
    data_dict = torch.load(d_path, weights_only=False)

    print(f"fraction value: {fraction}")

    full_samples = data_dict['train_images'].shape[0] 
    subset_size = int(full_samples * fraction) 

    new_total_images = data_dict['train_images'][:subset_size]
    new_total_labels_full = data_dict['train_labels'][:subset_size]

    total_samples = new_total_images.shape[0]
    print(f"Data size: {total_samples}")
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    train_data = new_total_images[:val_start]
    train_labels = new_total_labels_full[:val_start]
    val_data = new_total_images[val_start:]
    val_labels = new_total_labels_full[val_start:]
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader