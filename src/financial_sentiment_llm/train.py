"""Training utilities."""

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

def train_model(model, train_loader, val_loader, device='cuda', epochs=5, lr=2e-5, 
                patience=3, save_path=None):
    """Training loop with early stopping.
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: 'cuda' or 'cpu'
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Number of epochs to wait for improvement before stopping
        save_path: Path to save best model (optional)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}")
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model! (val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.4f})")
            
            # Optionally save checkpoint
            if save_path:
                checkpoint_path = Path(save_path).parent / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(best_model_state, checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best model was at epoch {best_epoch} "
                      f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
                
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    # If we completed all epochs without early stopping
    if patience_counter < patience and best_model_state is not None:
        print(f"\n✓ Training completed all {epochs} epochs")
        print(f"  Best model was at epoch {best_epoch} "
              f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
        model.load_state_dict(best_model_state)
    return model 