"""Multi-task training with dual loss."""

import torch
import torch.nn as nn
from tqdm import tqdm

def train_multi_task_model(model, train_loader, val_loader, device='cuda',  epochs=5, lr=2e-5, patience=3, classification_weight=1.0, regression_weight=10.0, save_path=None, debug=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.MSELoss()
    
    model.to(device)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        train_cls_batches = 0
        train_reg_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            task_types = batch['task_type']
            
            cls_mask = torch.tensor([t == 'classification' for t in task_types])
            reg_mask = torch.tensor([t == 'regression' for t in task_types])
            
            optimizer.zero_grad()
            total_loss = 0
            
            if cls_mask.any():
                cls_logits = model(input_ids[cls_mask], attention_mask[cls_mask], task_type='classification')
                # Cast to Long for CrossEntropy
                cls_loss = classification_loss_fn(cls_logits, targets[cls_mask].long())
                total_loss += classification_weight * cls_loss
                train_cls_batches += 1
            
            if reg_mask.any():
                reg_preds = model(input_ids[reg_mask], attention_mask[reg_mask], task_type='regression')
                reg_targets = targets[reg_mask].float()
                
                # ADD THIS BLOCK ===
                if epoch == 0 and train_reg_batches == 0:
                     if debug:
                        print(f"\nDEBUG: First batch regression targets: {reg_targets[:10].tolist()}")
                        print(f"DEBUG: First batch regression preds:   {reg_preds[:10].detach().cpu().tolist()}")
                # Cast to Float for MSE
                reg_loss = regression_loss_fn(reg_preds, targets[reg_mask].float())
                total_loss += regression_weight * reg_loss
                train_reg_batches += 1
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        cls_correct = 0
        cls_total = 0
        reg_mse_sum = 0
        reg_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                task_types = batch['task_type']
                
                cls_mask = torch.tensor([t == 'classification' for t in task_types])
                reg_mask = torch.tensor([t == 'regression' for t in task_types])
                
                batch_loss = 0
                
                if cls_mask.any():
                    cls_logits = model(input_ids[cls_mask], attention_mask[cls_mask], task_type='classification')
                    cls_targets = targets[cls_mask].long()
                    loss = classification_loss_fn(cls_logits, cls_targets)
                    batch_loss += classification_weight * loss
                    
                    # Accuracy metrics
                    predictions = torch.argmax(cls_logits, dim=1)
                    cls_correct += (predictions == cls_targets).sum().item()
                    cls_total += cls_targets.size(0)
                    
                if reg_mask.any():
                    reg_preds = model(input_ids[reg_mask], attention_mask[reg_mask], task_type='regression')
                    reg_targets = targets[reg_mask].float()
                    loss = regression_loss_fn(reg_preds, reg_targets)
                    batch_loss += regression_weight * loss
                    
                    # MSE metrics
                    reg_mse_sum += loss.item() * reg_targets.size(0) # weighted by batch size
                    reg_total += reg_targets.size(0)
                    
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Metrics strings
        val_cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        val_reg_mse = reg_mse_sum / reg_total if reg_total > 0 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        print(f"  > Classification Acc: {val_cls_acc:.4f} ({cls_total} samples)")
        print(f"  > Regression MSE:     {val_reg_mse:.4f} ({reg_total} samples)")
        
        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model!")
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stopping triggered.")
                model.load_state_dict(best_model_state)
                break
                
    return model
