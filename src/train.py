import torch
import os
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate(model, dataloader, device, loss_fn, model_type):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different model inputs
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            labels = batch['label'].to(device) # Class index for CrossEntropy
            
            if model_type == 'Lrgbd':
                outputs = model(rgb, depth)
            elif model_type == 'rgb':
                outputs = model(rgb)
            elif model_type == 'depth':
                outputs = model(depth)
            elif model_type == 'Ergbd':
                # Concatenate along channel dim (dim 1)
                inputs = torch.cat([rgb, depth], dim=1)
                outputs = model(inputs)
            
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_predictions) * 100
    return avg_loss, accuracy

def train_model(model, train_dl, val_dl, epochs, lr, device, model_type, save_dir, experiment_name):
    os.makedirs(save_dir, exist_ok=True)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_dl:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass based on type
            if model_type == 'Lrgbd':
                outputs = model(rgb, depth)
            elif model_type == 'rgb':
                outputs = model(rgb)
            elif model_type == 'depth':
                outputs = model(depth)
            elif model_type == 'Ergbd':
                inputs = torch.cat([rgb, depth], dim=1)
                outputs = model(inputs)
                
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        # Stats
        train_acc = (correct / total) * 100
        avg_train_loss = train_loss / len(train_dl)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_dl, device, loss_fn, model_type)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            
            # Save checkpoint
            save_path = os.path.join(save_dir, f"{experiment_name}_best.pth")
            torch.save(best_model_wts, save_path)
            print(f"--> New best model saved! ({val_acc:.2f}%)")
        else:
            no_improve_count += 1
            if no_improve_count >= 7: # Increased patience slightly
                print("Early stopping triggered.")
                break
                
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    model.load_state_dict(best_model_wts)
    return model