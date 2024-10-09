import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from offline_scripts.classifier_functions.LMDA_modified import LMDA
from offline_scripts.custom_functions.load_data import load_dataset
from offline_scripts.custom_functions.training_helpers import EarlyStopping


def train_model(train_loader: DataLoader, val_loader: DataLoader, config: dict, device, fold: int = 0):
    # criterion_l1 = torch.nn.L1Loss().cuda()
    # criterion_l2 = torch.nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda(device=device)
    model = LMDA(num_classes=len(config["data"]["selected_classes"]),
                 chans=config["model"]["lmda"]["chans"],
                 samples=config["model"]["lmda"]["samples"],
                 channel_depth1=config["model"]["lmda"]["channel_depth1"],
                 channel_depth2=config["model"]["lmda"]["channel_depth2"]).to(device)
    if config["model"]["load_model"]:
        model.load_state_dict(torch.load(f'{config["model"]["path"]}/best_LMDA_PhS_fold-{fold}.pt', weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

    train_loss_epoch = torch.zeros(config["training"]["epochs"])  # the train loss for each epoch
    val_loss_epoch = torch.zeros(config["training"]["epochs"])  # the val loss for each epoch
    train_acc_epoch = torch.zeros(config["training"]["epochs"])  # the train acc for each epoch
    val_acc_epoch = torch.zeros(config["training"]["epochs"])  # the val acc for each epoch

    early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"],
                                   min_delta=config["training"]["early_stopping"]["min_delta"])

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.max(outputs, 1)[1]
            train_acc += float((y_pred == labels).to(device).numpy().astype(int).sum()) / float(labels.size(0))
            train_loss += loss.item()

        scheduler.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_loss_epoch[epoch] = train_loss
        train_acc_epoch[epoch] = train_acc

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                y_pred = torch.max(outputs, 1)[1]
                val_acc += float((y_pred == labels).to(device).numpy().astype(int).sum()) / float(labels.size(0))
                val_loss += loss.item()


        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_loss_epoch[epoch] = val_loss
        val_acc_epoch[epoch] = val_acc

        wandb.run.log({
            f"{fold}/train/accuracy": train_acc,
            f"{fold}/train/loss": train_loss,
            f"{fold}/validation/accuracy": val_acc,
            f"{fold}/validation/loss": val_loss,
            f'{fold}/best/acc': torch.max(val_acc_epoch),
        })

        print('Fold:', fold,
              'Epoch:', epoch + 1,
              '  Train loss: %.6f' % train_loss,
              '  Val loss: %.6f' % val_loss,
              '  Train accuracy %.6f' % train_acc,
              '  Val accuracy is %.6f' % val_acc,
              '  Best accuracy is %.6f' % torch.max(val_acc_epoch))

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            return model, val_acc_epoch[epoch]

        # Save the best model
        if not early_stopping.early_stop:
            torch.save(model.state_dict(), f'{config["model"]["path"]}/best_LMDA_PhS_fold-{fold}.pt')

    return model, early_stopping.acc

def load_data(x, y, device, config: dict, shuffle: bool = False) -> torch.utils.data.DataLoader:
    data = np.expand_dims(x, axis=1)
    img = (torch.from_numpy(data).to(device)).type(torch.float32)
    label = (torch.from_numpy(y).to(device)).type(torch.long)
    dataset = torch.utils.data.TensorDataset(img, label)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle)

def evaluate(device, config: dict):
    X, y = load_dataset(config)

    kf = KFold(n_splits=config["training"]["k_fold_splits"])
    acc_list = []

    fold = 1
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dataloader = load_data(x=X_train, y=y_train, device=device, config=config, shuffle=True)
        test_dataloader = load_data(x=X_test, y=y_test, device=device, config=config)

        _, acc = train_model(train_loader=dataloader, val_loader=test_dataloader, config=config, device=device,
                             fold=fold)
        acc_list.append(acc)
        if device == "cuda":
            torch.cuda.empty_cache()


        fold += 1
    print(f'Average accuracy over all folds: {np.mean(acc_list)}')

def train_final(device, config: dict):
    X, y = load_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["final"]["test_split"],
                                                        random_state=0)
    dataloader = load_data(x=X_train, y=y_train, device=device, config=config, shuffle=True)
    test_dataloader = load_data(x=X_test, y=y_test, device=device, config=config)

    model, acc = train_model(train_loader=dataloader, val_loader=test_dataloader, config=config, device=device)

    print(f'Final acc: {acc}')
    return