import torch
import os
import os.path as osp
import json
import typer
import numpy as np
import pickle
import datetime 
import joblib

from models.encoder import GCNN
from utils import preprocess, train_rf_classifier, train_rf_regressor, train_cycle_classifier, train_cycle_regressor, get_battery_loaders


def main_old(dataset_name: str,
         experiment_name: str = typer.Argument("test"),
         lr: float = typer.Option(0.01),
         hidden_size: int = typer.Option(32),
         batch_size: int = typer.Option(32),
         dropout: float = typer.Option(0.1),
         epochs:int = typer.Option(50),
         seed: int = typer.Option(0)):

    torch.manual_seed(seed)

    dataset_name = dataset_name.lower()

    base_path = './runs_meg/' + dataset_name  + '/' + experiment_name
    if not osp.exists(base_path):
        os.makedirs(base_path + "/ckpt")
        os.makedirs(base_path + "/plots")
        os.makedirs(base_path + "/splits")
        os.makedirs(base_path + "/meg_output")
    else:
        import shutil
        shutil.rmtree(base_path + "/plots", ignore_errors=True)
        os.makedirs(base_path + "/plots")


    train_loader, val_loader, test_loader, *extra = preprocess(dataset_name, experiment_name, batch_size)
    train_ds, val_ds, test_ds, num_features, num_classes = extra

    len_train = len(train_ds)
    len_val = len(val_ds)
    len_test = len(test_ds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCNN(
        num_input=num_features,
        num_hidden=hidden_size,
        num_output=num_classes,
        dropout=dropout
    ).to(device)

    with open(base_path + '/hyperparams.json', 'w') as outfile:
        json.dump({'num_input': num_features,
                   'num_hidden': hidden_size,
                   'num_output': num_classes,
                   'dropout': dropout,
                   'seed': seed}, outfile)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    if dataset_name.lower() in ['tox21', 'cycliq', 'cycliq-multi']:
        train_cycle_classifier(task=dataset_name.lower(),
                               train_loader=train_loader,
                               val_loader=val_loader,
                               test_loader=test_loader,
                               len_train=len_train,
                               len_val=len_val,
                               len_test=len_test,
                               model=model,
                               optimizer=optimizer,
                               device=device,
                               base_path=base_path,
                               epochs=epochs)

    elif dataset_name.lower() in ['esol']:
        train_cycle_regressor(task=dataset_name.lower(),
                              train_loader=train_loader,
                              val_loader=val_loader,
                              test_loader=test_loader,
                              len_train=len_train,
                              len_val=len_val,
                              len_test=len_test,
                              model=model,
                              optimizer=optimizer,
                              device=device,
                              base_path=base_path,
                              epochs=epochs)
        
def main(data_file: str,
         dataset_name: str,
         experiment_name: str = typer.Argument("test"),
         n_estimators: int = typer.Option(100),
         max_depth: int = typer.Option(None),
         batch_size: int = typer.Option(32),
         folds: int = typer.Option(5),
         seed: int = typer.Option(0)):

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_name = dataset_name.lower()
    
    base_path = osp.join(os.getcwd(), 'RFReg', experiment_name)
    
    if not osp.exists(base_path):
        os.makedirs(base_path + "/ckpt")
        os.makedirs(base_path + "/plots")
        # os.makedirs(base_path + "/splits")
        # os.makedirs(base_path + "/meg_output")
    else:
        import shutil
        shutil.rmtree(base_path + "/plots", ignore_errors=True)
        os.makedirs(base_path + "/plots")

    # train_loader, val_loader, test_loader, *extra = preprocess(data_file, dataset_name, experiment_name, batch_size, folds, seed)
    preprocess(data_file, dataset_name, experiment_name, batch_size, folds, seed)
    for f in range(folds):
        train_loader, val_loader, test_loader, *extra = get_battery_loaders(experiment_name, batch_size, f)
        train_ds, val_ds, test_ds, num_features, num_classes = extra

        len_train = len(train_ds)
        len_val = len(val_ds)
        len_test = len(test_ds)

        device = torch.device('cpu') # rf doesn't use GPU
        
        with open(base_path + f'/ckpt/hyperparams{f}.json', 'w') as outfile:
            json.dump({
                'model': 'RandomForest',
                'n_estimators': n_estimators,
                'max_depth': max_depth if max_depth else "None",
                'num_input': num_features,
                'num_output': num_classes,
                'batch_size': batch_size,
                'seed': seed,
                'device': device.type
            }, outfile)

        if dataset_name.lower() in ['tox21', 'cycliq', 'cycliq-multi']:
            rf_model, results = train_rf_classifier(
                task=dataset_name.lower(),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                len_train=len_train,
                len_val=len_val,
                len_test=len_test,
                base_path=base_path,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed
            )
            with open(base_path + '/ckpt/rf_classifier_model.pkl', 'wb') as file:
                pickle.dump(rf_model, file)

        elif dataset_name.lower() in ['esol', 'battery']:
            rf_model, results = train_rf_regressor(
                task=dataset_name.lower(),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                len_train=len_train,
                len_val=len_val,
                len_test=len_test,
                base_path=base_path,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                fold_num=f
            )
            
            # Save using joblib (more efficient for sklearn models)
            joblib.dump(rf_model, base_path + f'/ckpt/model_{f}.joblib')
        
            # with open(base_path + '/ckpt/rf_regressor_model.pkl', 'wb') as f:
            #     pickle.dump(rf_model, f)
                
            # Save feature information to ensure consistency during prediction
            feature_info = {
                "num_features": rf_model.n_features_in_,
                "feature_details": "MACCS fingerprints only (no ID column)",
                "last_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": {
                    "joblib": base_path + f'/ckpt/model_{f}.joblib'
                    # "pickle": base_path + '/ckpt/rf_regressor_model.pkl'
                }
            }
            
            with open(base_path + f'/ckpt/feature_info_{f}.json', 'w') as file:
                json.dump(feature_info, file, indent=2)


if __name__ == '__main__':
    typer.run(main)
