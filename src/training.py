import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5, model_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.model_path = model_path
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.model_path)


def setup_training(use_npc=False, learning_rate=0.0001, weight_decay=0.0001):
    
    # loss: either MSE or NPC
    if use_npc:
        criterion = NegativePearsonCorrelation()
    else:
        criterion = nn.MSELoss()

    # adam optimizer 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # early stopping 
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    
    return criterion, optimizer, scheduler, early_stopping


def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)  
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs)  
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Training for DeepPhys and PhysNet                                                 '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                epochs=20, 
                device='cuda', 
                early_stopping=None):
    
    train_losses = []
    val_losses = []
    
    # Create the loss curves plot for W&B
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    
    for epoch in range(epochs):
        # ---------------- TRAINING ----------------
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truth = []
        print(f"Epoch {epoch + 1}/{epochs}")
        train_progress = tqdm(train_loader, desc="Training")

        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
            
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_ground_truth.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_ground_truth = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_ground_truth.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss) # Learning rate scheduling

        # loss curves plot
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(range(1, epoch + 2), train_losses, label='Training Loss', marker='o')
        ax_loss.plot(range(1, epoch + 2), val_losses, label='Validation Loss', marker='o')
        ax_loss.set_title('Training and Validation Loss Curves')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # prediction vs actual 
        fig_pred, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(train_predictions[:100], label='Predictions', alpha=0.7)
        ax1.plot(train_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax1.set_title('Training: Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_predictions[:100], label='Predictions', alpha=0.7)
        ax2.plot(val_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax2.set_title('Validation: Predictions vs Ground Truth')
        ax2.legend()
        ax2.grid(True)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "loss_curves": wandb.Image(fig_loss),
            "predictions_vs_ground_truth": wandb.Image(fig_pred)
        })

        plt.close(fig_loss)
        plt.close(fig_pred)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if early_stopping is not None: # Early stopping check
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    wandb.run.summary.update({
        "best_val_loss": min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_epochs": len(train_losses),
        "early_stopped": early_stopping.early_stop if early_stopping else False,
    })

    return train_losses, val_losses


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Training for sequential CNN (without BVP)                                        '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def train_cnnlstm(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler,  
    epochs = 20, 
    device ='cuda',
    early_stopping=None  
):
    
    train_losses = []
    val_losses = []

    # Create the loss curves plot for W&B
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")


    for epoch in range(epochs):
        # ---------------- TRAINING ----------------
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truth = []
        print(f"Epoch {epoch + 1}/{epochs}")
        train_progress = tqdm(train_loader, desc="Training")

        for frames, bvp, eda in train_progress:
            frames = frames.to(device)
            eda = eda.to(device)  # We’re predicting EDA

            optimizer.zero_grad()
            outputs = model(frames)     # (B, ...)
            outputs = outputs.squeeze(-1)  # If shape was (B, T, 1) -> (B, T), or (B,1)->(B,)

            loss = criterion(outputs, eda)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * frames.size(0)
            train_progress.set_postfix(loss=loss.item())
            
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_ground_truth.extend(eda.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_ground_truth = []

        with torch.no_grad():
            for frames, bvp, eda in tqdm(val_loader, desc="Validation"):
                frames = frames.to(device)
                eda = eda.to(device)

                outputs = model(frames)
                outputs = outputs.squeeze(-1)

                loss = criterion(outputs, eda)
                val_loss += loss.item() * frames.size(0)
                val_predictions.extend(outputs.cpu().numpy())
                val_ground_truth.extend(eda.cpu().numpy())

              
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss) # Learning rate scheduling

         # loss curves plot
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(range(1, epoch + 2), train_losses, label='Training Loss', marker='o')
        ax_loss.plot(range(1, epoch + 2), val_losses, label='Validation Loss', marker='o')
        ax_loss.set_title('Training and Validation Loss Curves')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # prediction vs actual 
        fig_pred, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(train_predictions[:100], label='Predictions', alpha=0.7)
        ax1.plot(train_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax1.set_title('Training: Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_predictions[:100], label='Predictions', alpha=0.7)
        ax2.plot(val_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax2.set_title('Validation: Predictions vs Ground Truth')
        ax2.legend()
        ax2.grid(True)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "loss_curves": wandb.Image(fig_loss),
            "predictions_vs_ground_truth": wandb.Image(fig_pred)
        })

        plt.close(fig_loss)
        plt.close(fig_pred)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if early_stopping is not None: # Early stopping check
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
    wandb.run.summary.update({
        "best_val_loss": min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_epochs": len(train_losses),
        "early_stopped": early_stopping.early_stop if early_stopping else False,
    })

    return train_losses, val_losses


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Training for sequential CNN + BVP                                                '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def train_cnnlstm_bvp(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler,  
    epochs = 20, 
    device ='cuda',
    early_stopping=None  
):
    
    train_losses = []
    val_losses = []

    # Create the loss curves plot for W&B
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")


    for epoch in range(epochs):
        # ---------------- TRAINING ----------------
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truth = []
        print(f"Epoch {epoch + 1}/{epochs}")
        train_progress = tqdm(train_loader, desc="Training")

        for sequences, bvp_values, eda_targets in train_progress:
            sequences = sequences.to(device)
            bvp_values = bvp_values.to(device)
            eda_targets = eda_targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, bvp_values)
            outputs = outputs.squeeze(-1)  # shape might be (B, T, 1), so -> (B, T)

            loss = criterion(outputs, eda_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            train_progress.set_postfix(loss=loss.item())
            
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_ground_truth.extend(eda_targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_ground_truth = []

        with torch.no_grad():

            for sequences, bvp_values, eda_targets in tqdm(val_loader, desc="Validation"):
                sequences = sequences.to(device)
                bvp_values = bvp_values.to(device)
                eda_targets = eda_targets.to(device)

                # forward
                outputs = model(sequences, bvp_values)
                outputs = outputs.squeeze(-1)

                # loss
                loss = criterion(outputs, eda_targets)
                val_loss += loss.item() * sequences.size(0)

                val_predictions.extend(outputs.cpu().numpy())
                val_ground_truth.extend(eda_targets.cpu().numpy())
              
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss) # Learning rate scheduling

         # loss curves plot
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(range(1, epoch + 2), train_losses, label='Training Loss', marker='o')
        ax_loss.plot(range(1, epoch + 2), val_losses, label='Validation Loss', marker='o')
        ax_loss.set_title('Training and Validation Loss Curves')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # prediction vs actual 
        fig_pred, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(train_predictions[:100], label='Predictions', alpha=0.7)
        ax1.plot(train_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax1.set_title('Training: Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_predictions[:100], label='Predictions', alpha=0.7)
        ax2.plot(val_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax2.set_title('Validation: Predictions vs Ground Truth')
        ax2.legend()
        ax2.grid(True)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "loss_curves": wandb.Image(fig_loss),
            "predictions_vs_ground_truth": wandb.Image(fig_pred)
        })

        plt.close(fig_loss)
        plt.close(fig_pred)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if early_stopping is not None: # Early stopping check
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
    wandb.run.summary.update({
        "best_val_loss": min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_epochs": len(train_losses),
        "early_stopped": early_stopping.early_stop if early_stopping else False,
    })

    return train_losses, val_losses

