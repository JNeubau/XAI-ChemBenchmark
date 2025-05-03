import torch
import torch.nn.functional as F
import os.path as osp
import json

from torch_geometric.utils import precision, recall
from torch_geometric.utils import f1_score, accuracy
from torch.utils.tensorboard import SummaryWriter

import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def train_epoch_classifier(model, train_loader, len_train, optimizer, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train

def test_classifier(model, loader, device):
    model.eval()

    y = torch.tensor([]).long().to(device)
    yp = torch.tensor([]).long().to(device)

    loss_all = 0
    for data in loader:
        data = data.to(device)
        pred, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(F.log_softmax(pred, dim=-1), data.y)
        pred = pred.max(dim=1)[1]

        y = torch.cat([y, data.y])
        yp = torch.cat([yp, pred])

        loss_all += data.num_graphs * loss.item()

    return (
        accuracy(y, yp),
        precision(y, yp, model.num_output).mean().item(),
        recall(y, yp, model.num_output).mean().item(),
        f1_score(y, yp, model.num_output).mean().item(),
        loss_all
    )

def train_cycle_classifier(task, train_loader, val_loader, test_loader, len_train, len_val, len_test,
                           model, optimizer, device, base_path, epochs):

    best_acc = (0, 0)
    writer = SummaryWriter(base_path + '/plots')

    for epoch in range(epochs):
        loss = train_epoch_classifier(model, train_loader, len_train, optimizer, device)
        writer.add_scalar('Loss/train', loss, epoch)
        train_acc, train_prec, train_rec, train_f1, _ = test_classifier(model, train_loader, device)
        val_acc, val_prec, val_rec, val_f1, l = test_classifier(model, val_loader, device)

        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Loss/val', l / len_val, epoch)

        print(f'Epoch: {epoch}, Loss: {loss:.5f}')

        print(f'Train -> Acc: {train_acc:.5f}  Rec: {train_rec:.5f}  \
        Prec: {train_prec:.5f}  F1: {train_f1:.5f}')

        print(f'Val -> Acc: {val_acc:.5f}  Rec: {val_rec:.5f}  \
        Prec: {val_prec:.5f}  F1: {val_f1:.5f}')

        if best_acc[1] < val_acc:
            best_acc = train_acc, val_acc

            torch.save(
                model.state_dict(),
                osp.join(base_path + '/ckpt/',
                         model.__class__.__name__ + ".pth")
            )
            print("New best model saved!")

            with open(base_path + '/best_result.json', 'w') as outfile:
                json.dump({'train_acc': train_acc,
                           'val_acc': val_acc,
                           'train_rec': train_rec,
                           'val_rec': val_rec,
                           'train_f1': train_f1,
                           'val_f1': val_f1,
                           'train_prec': train_prec,
                           'val_prec': val_prec}, outfile)


def train_epoch_regressor(model, train_loader, len_train, optimizer, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(output, data.y)

        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train


def test_regressor(model, loader, len_loader, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)

        pred, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(pred, data.y).detach()

        loss_all += data.num_graphs * loss.item()

    return loss_all / len_loader


def train_cycle_regressor(task, train_loader, val_loader, test_loader,
                          len_train, len_val, len_test, model,
                          optimizer, device, base_path, epochs):

    best_acc = (0, 0)
    writer = SummaryWriter(base_path + '/plots')

    best_error = (+10000, +10000)
    for epoch in range(epochs):
        loss = train_epoch_regressor(model, train_loader, len_train, optimizer, device)
        writer.add_scalar('Loss/train', loss, epoch)
        train_error = test_regressor(model, train_loader, len_train, device)
        val_error = test_regressor(model, val_loader, len_val, device)

        writer.add_scalar('MSE/train', train_error, epoch)
        writer.add_scalar('MSE/test', val_error, epoch)

        print(f'Epoch: {epoch}, Loss: {loss:.5f}')

        print(f'Training Error: {train_error:.5f}')
        print(f'Val Error: {val_error:.5f}')

        if best_error[1] > val_error:
            best_error = train_error, val_error
            torch.save(
                model.state_dict(),
                osp.join(base_path + '/ckpt/',
                         model.__class__.__name__ + ".pth")
            )
            print("New best model saved!")

            with open(base_path + '/best_result.json', 'w') as outfile:
                json.dump({'train_error': train_error,
                           'val_error': val_error}, outfile)

def train_rf_classifier(task, train_loader, val_loader, test_loader, len_train, len_val, len_test, 
                        base_path, n_estimators=100, max_depth=None, random_state=0):
    # Extract features and labels from data loaders
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for data in train_loader:
        # Assuming data has features in data.x and labels in data.y
        # Flatten graph features to use with Random Forest
        batch_features = []
        for i in range(len(data.y)):
            # Get node features for this graph and flatten them
            mask = data.batch == i
            graph_features = data.x[mask].view(-1).numpy()
            batch_features.append(graph_features)
        
        X_train.extend(batch_features)
        y_train.extend(data.y.numpy())
    
    for data in val_loader:
        batch_features = []
        for i in range(len(data.y)):
            mask = data.batch == i
            graph_features = data.x[mask].view(-1).numpy()
            batch_features.append(graph_features)
        
        X_val.extend(batch_features)
        y_val.extend(data.y.numpy())
        
    for data in test_loader:
        batch_features = []
        for i in range(len(data.y)):
            mask = data.batch == i
            graph_features = data.x[mask].view(-1).numpy()
            batch_features.append(graph_features)
        
        X_test.extend(batch_features)
        y_test.extend(data.y.numpy())
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Predictions
    train_pred = rf.predict(X_train)
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)
    
    # Probabilities for ROC AUC
    train_proba = rf.predict_proba(X_train)[:, 1]
    val_proba = rf.predict_proba(X_val)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    
    # Print results
    print(f"Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    # Save results
    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc
    }
    
    with open(base_path + '/rf_classifier_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(min(20, len(rf.feature_importances_))), 
             rf.feature_importances_[:20] if len(rf.feature_importances_) > 20 else rf.feature_importances_)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(base_path + '/plots/rf_feature_importance.png')
    
    return rf, results


def train_rf_regressor(task, train_loader, val_loader, test_loader, len_train, len_val, len_test, 
                       base_path, n_estimators=100, max_depth=None, random_state=0):
    # Extract features and labels from data loaders
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for data in train_loader:
        batch_features = []
        for i in range(len(data.y)):
            mask = data.batch == i
            graph_features = data.x[mask].view(-1).numpy()
            batch_features.append(graph_features)
        
        X_train.extend(batch_features)
        y_train.extend(data.y.numpy())
    
    for data in val_loader:
        batch_features = []
        for i in range(len(data.y)):
            mask = data.batch == i
            graph_features = data.x[mask].view(-1).numpy()
            batch_features.append(graph_features)
        
        X_val.extend(batch_features)
        y_val.extend(data.y.numpy())
        
    for data in test_loader:
        batch_features = []
        for i in range(len(data.y)):
            mask = data.batch == i
            graph_features = data.x[mask].view(-1).numpy()
            batch_features.append(graph_features)
        
        X_test.extend(batch_features)
        y_test.extend(data.y.numpy())
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Predictions
    train_pred = rf.predict(X_train)
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Print results
    print(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Save results
    results = {
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
        "test_r2": float(test_r2)
    }
    
    with open(base_path + '/rf_regressor_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(min(20, len(rf.feature_importances_))), 
             rf.feature_importances_[:20] if len(rf.feature_importances_) > 20 else rf.feature_importances_)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(base_path + '/plots/rf_feature_importance.png')
    
    return rf, results