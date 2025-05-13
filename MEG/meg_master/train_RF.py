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
from utils import preprocess, train_rf_classifier, train_rf_regressor, train_cycle_classifier, train_cycle_regressor


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
        
def main(dataset_name: str,
         experiment_name: str = typer.Argument("test"),
         n_estimators: int = typer.Option(100),
         max_depth: int = typer.Option(None),
         batch_size: int = typer.Option(32),
         seed: int = typer.Option(0)):

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_name = dataset_name.lower()
    
    work_dir = os.getcwd()
    base_path = osp.join(work_dir, 'runs_meg', dataset_name, experiment_name)
    
    if not osp.exists(base_path):
        os.makedirs(base_path + "/ckpt")
        os.makedirs(base_path + "/plots")
        os.makedirs(base_path + "/splits")
        os.makedirs(base_path + "/meg_output")
    else:
        import shutil
        shutil.rmtree(base_path + "/plots", ignore_errors=True)
        os.makedirs(base_path + "/plots")

    train_loader, val_loader, test_loader, *extra = preprocess(dataset_name, experiment_name, batch_size, seed)
    train_ds, val_ds, test_ds, num_features, num_classes = extra

    len_train = len(train_ds)
    len_val = len(val_ds)
    len_test = len(test_ds)

    device = torch.device('cpu') # rf doesn't use GPU
    
    with open(base_path + '/hyperparams.json', 'w') as outfile:
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
        with open(base_path + '/ckpt/rf_classifier_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

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
            random_state=seed
        )
        
        # Save using joblib (more efficient for sklearn models)
        joblib.dump(rf_model, base_path + '/ckpt/model.joblib')
    
        with open(base_path + '/ckpt/rf_regressor_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
            
        # Save feature information to ensure consistency during prediction
        feature_info = {
            "num_features": rf_model.n_features_in_,
            "feature_details": "MACCS fingerprints only (no ID column)",
            "last_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": {
                "joblib": base_path + '/ckpt/rf_regressor_model.joblib',
                "pickle": base_path + '/ckpt/rf_regressor_model.pkl'
            }
        }
        
        with open(base_path + '/ckpt/feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
    # elif dataset_name.lower() in ['battery']:
    #     rf_model, results = train_rf_regressor(
    #         task=dataset_name.lower(),
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         test_loader=test_loader,
    #         len_train=len_train,
    #         len_val=len_val,
    #         len_test=len_test,
    #         base_path=base_path,
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         random_state=seed
    #     )
    #     with open(base_path + '/ckpt/rf_regressor_model.pkl', 'wb') as f:
    #         pickle.dump(rf_model, f)


if __name__ == '__main__':
    typer.run(main)
