import torch
import torch.nn as nn
import numpy as np
import wandb

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Testing for DeepPhys and PhysNet                                                 '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def test_model(model, test_loader, criterion, device='cuda', dual_input=False):
    """
    Test the model on the test set, supporting both single and dual input models.
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for batch in test_loader:
            if dual_input:
                inputs, labels, extra_input = batch
                inputs, labels, extra_input = inputs.to(device), labels.to(device), extra_input.to(device)
                outputs = model(inputs, extra_input)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    wandb.log({"test loss": avg_test_loss})
    data = [[i, float(t), float(p)] for i, (t, p) in enumerate(zip(all_labels, all_preds))]
    table = wandb.Table(data=data, columns=["index", "actual", "prediction"])
    wandb.log({"test_predictions_table": table})

    return avg_test_loss, all_preds, all_labels



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Testing for sequential CNN (without BVP)                                         '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def test_cnnlstm(model, 
                test_loader, 
                criterion, 
                device, 
                log_predictions=False):

    model.eval()
    test_loss = 0.0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch_idx, (frames, bvp, eda) in enumerate(test_loader):
            
            frames = frames.to(device)  
            eda = eda.to(device)        

            outputs = model(frames)     
            outputs = outputs.squeeze(-1)  
    
            loss = criterion(outputs, eda)
            test_loss += loss.item()

            predictions.extend(outputs.cpu().numpy().flatten())
            ground_truth.extend(eda.cpu().numpy().flatten())

    
    test_loss_avg = test_loss / len(test_loader.dataset)

    predictions_np = np.array(predictions)
    ground_truth_np = np.array(ground_truth)
    mae = np.mean(np.abs(predictions_np - ground_truth_np))
    rmse = np.sqrt(np.mean((predictions_np - ground_truth_np) ** 2))

    test_metrics = {
        "Test Loss": test_loss_avg,
        "MAE": mae,
        "RMSE": rmse
    }

    print(f"Test Loss: {test_loss_avg:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return test_loss_avg, test_metrics, predictions, ground_truth



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Testing for sequential CNN + BVP                                                 '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def test_cnnlstm_bvp(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    
    true_eda = []
    pred_eda = []
    
    with torch.no_grad():
        for sequences, eda_targets, bvp_values in test_loader:

            sequences = sequences.to(device)
            eda_targets = eda_targets.to(device)
            bvp_values = bvp_values.to(device)

            outputs = model(sequences, bvp_values)  # => (B, T, 1)
            outputs = outputs.squeeze(-1)          # => (B, T)

            if len(eda_targets.shape) == 1:
                eda_targets = eda_targets.unsqueeze(-1).repeat(1, outputs.size(1))

            loss = criterion(outputs, eda_targets)
            test_loss += loss.item()

            true_eda.extend(eda_targets.cpu().numpy().flatten())
            pred_eda.extend(outputs.cpu().numpy().flatten())

    test_loss /= len(test_loader)

    return test_loss, true_eda, pred_eda