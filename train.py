"""
MAI/IDL SS26 - Final assignment. 
MG 6/6/2026
"""
import json
import time
import psutil
import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer


def track_efficiency(model, val_loader, device):
    """Measure inference speed and peak memory during inference"""
    model.eval()
    total_samples = 0
    inference_memory_peak = 0
    
    process = psutil.Process(os.getpid())       
    
    inference_start = time.time()
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            _ = model(x)
            total_samples += x.shape[0]
            
            # Track memory during inference
            memory_gb = process.memory_info().rss / 1e9
            inference_memory_peak = max(inference_memory_peak, memory_gb)
     
    inference_time = time.time() - inference_start
    inference_latency = (inference_time / total_samples) * 1000  # ms per image
    
    return inference_latency, inference_memory_peak


def save_efficiency_results(model_name, dataset_name, total_time, inference_latency, peak_memory):
    """Save efficiency metrics to CSV"""
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_name,
        'dataset': dataset_name,
        'total_training_time_sec': f"{total_time:.2f}",
        'inference_latency_ms': f"{inference_latency:.3f}",
        'peak_memory_gb': f"{peak_memory:.2f}"
    }
    
    filename = 'efficiency_results.csv'
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        fieldnames = results.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)
    
    return results


def print_efficiency_results(model_name, dataset_name, total_time, inference_latency, peak_memory):
    """Print efficiency metrics to console"""
    print("\n" + "="*70)
    print("EFFICIENCY TRACKING RESULTS (Part 2: Green Initiative)")
    print("="*70)
    print(f"Model:                      {model_name}")
    print(f"Dataset:                    {dataset_name}")
    print(f"Total Training Time:        {total_time:.2f} seconds")
    print(f"Inference Latency:          {inference_latency:.3f} ms per image")
    print(f"Peak Memory:                {peak_memory:.2f} GB")
    print("="*70 + "\n")


def main():   
    with open("./config.json", "r") as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # Device agnostic
    print(f"Training executing on device: {device}")
    
    train_loader, val_loader, _ = get_loaders(
        data=config["DATA"], 
        data_path=config["DATA_PATH"], 
        batch_size=config["BATCH_SIZE"]
    )
    
    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=config["CHANNELS"], 
        num_classes=config["NUM_CLASSES"], 
        drop_rate=0.50, 
        activation_str=None
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    trainer = Trainer(model, criterion, optimizer, device)
    
    # ========== EFFICIENCY TRACKING: START ==========
    process = psutil.Process(os.getpid())
    training_start = time.time()
    peak_memory_training = 0
    
    # ========== TRAINING ==========
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    
    # ========== EFFICIENCY TRACKING: DURING TRAINING ==========
    training_end = time.time()
    total_training_time = training_end - training_start
    peak_memory_training = process.memory_info().rss / 1e9      # Devided by 1e9 bcs of bytes formate of billion
    
    # ========== EFFICIENCY TRACKING: INFERENCE ==========
    print("\nMeasuring inference efficiency...")
    inference_latency, inference_memory_peak = track_efficiency(model, val_loader, device)
    
    # Peak memory is the max of training and inference
    peak_memory_overall = max(peak_memory_training, inference_memory_peak)
    
    # ========== SAVE & PRINT RESULTS ==========
    results = save_efficiency_results(
        model_name=config["MODEL"],
        dataset_name=config["DATA"],
        total_time=total_training_time,
        inference_latency=inference_latency,
        peak_memory=peak_memory_overall
    )
    
    print_efficiency_results(
        model_name=config["MODEL"],
        dataset_name=config["DATA"],
        total_time=total_training_time,
        inference_latency=inference_latency,
        peak_memory=peak_memory_overall
    )


if __name__ == "__main__":
    main()