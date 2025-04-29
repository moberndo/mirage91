import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from offline_scripts.classifier_functions.LMDA_modified import LMDA

def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters())

def train_model(dataloader: DataLoader, test_dataloader: DataLoader, config: dict, device):
    # criterion_l1 = torch.nn.L1Loss().cuda()
    # criterion_l2 = torch.nn.MSELoss().cuda()
    criterion_cls = torch.nn.CrossEntropyLoss().cuda(device=device)
    model = LMDA(num_classes=len(config["data"]["selected_classes"]), chans=32, samples=267, channel_depth1=24,
                 channel_depth2=7).to(device)
    if config["model"]["load_model"]:
        model.load_state_dict(torch.load(config["model"]["path"], weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

    best_acc = 0
    average_acc = 0
    num = 0
    acc_list = []

    train_loss = torch.zeros(config["training"]["epochs"])
    test_loss = torch.zeros(config["training"]["epochs"])

    for e in range(config["training"]["epochs"]):
        # in_epoch = time.time()
        model.train()
        loss = []
        train_acc = []
        for i, (img, label) in enumerate(dataloader):
            outputs = model(img)
            train_pred = torch.max(outputs, 1)[1]
            train_acc.append(
                float((train_pred == label).to(device).numpy().astype(int).sum()) / float(label.size(0)))
            loss_grad = criterion_cls(outputs, label)
            loss.append(loss_grad)

            optimizer.zero_grad()
            loss_grad.backward()
            optimizer.step()

        loss = torch.tensor(loss, dtype=torch.float)
        train_acc = torch.tensor(train_acc, dtype=torch.float)
        train_acc = torch.mean(train_acc)
        loss = torch.mean(loss)

        # test process
        if (e + 1) % 1 == 0:
            model.eval()
            loss_test = []
            acc = []
            with torch.no_grad():
                for j, (img_t, label_t) in enumerate(test_dataloader):
                    Cls = model(img_t)

                    loss_test.append(criterion_cls(Cls, label_t))
                    y_pred = (torch.max(Cls, 1)[1])
                    acc.append(float((y_pred == label_t).cpu().sum().numpy()) / float(label_t.size(0)))

                loss_test = torch.tensor(loss_test, dtype=torch.float)
                acc = torch.tensor(acc, dtype=torch.float)
                acc = torch.mean(acc)
                loss_test = torch.mean(loss_test)

                train_loss[e] = loss
                test_loss[e] = loss_test
                acc_list.append(acc)

            print('Epoch:', e,
                  '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                  '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                  '  Train accuracy %.6f' % train_acc,
                  '  Test accuracy is %.6f' % acc,
                  '  Best accuracy is %.6f' % best_acc)

            scheduler.step()
            # log_write.write(str(e) + "    " + str(acc) + "\n")
            num = num + 1
            average_acc = average_acc + acc
            if acc > best_acc:
                best_acc = acc
    return model, acc_list[-1]

def load_data(x, y, device, config: dict, shuffle: bool = False) -> torch.utils.data.DataLoader:
    data = np.expand_dims(x, axis=1)
    img = (torch.from_numpy(data).to(device)).type(torch.float32)
    label = (torch.from_numpy(y).to(device)).type(torch.long)
    dataset = torch.utils.data.TensorDataset(img, label)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle)

def evaluate(device, config: dict):
    X = np.load(config["data"]["dataset_path"], allow_pickle=True)

    labels = (np.array(X[:, 0])).astype(np.float32)
    X = np.array(X[:, 1].tolist())
    Label = labels

    X = X[:, :, 200 * 2:200 * 6]
    X = X[:, :, 0:-1:3]

    combined_indices = np.where(np.isin(Label, config["data"]["selected_classes"]))[0]
    # Use the combined indices to index X and Label
    X = X[combined_indices, :, :]
    Label = Label[combined_indices]

    c = np.random.permutation(X.shape[0])
    X = X[c, :, :]
    Label = Label[c]

    # Get unique values and create a mapping
    unique_values = np.unique(Label)
    mapping = {val: idx for idx, val in enumerate(unique_values)}

    # Apply the mapping to the Label array
    Label = np.array([mapping[val] for val in Label])

    kf = KFold(n_splits=config["training"]["k_fold_splits"])
    acc_list = []

    fold = 1
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Label[train_index], Label[test_index]

        dataloader = load_data(x=X_train, y=y_train, device=device, config=config, shuffle=True)
        test_dataloader = load_data(x=X_test, y=y_test, device=device, config=config)

        _, acc = train_model(dataloader=dataloader, test_dataloader=test_dataloader, config=config, device=device)
        acc_list.append(acc)
        if device == "cuda":
            torch.cuda.empty_cache()


        fold += 1
    print(f'Average accuracy over all folds: {np.mean(acc_list)}')

def train_final(device, config: dict):
    X = np.load(config["data"]["dataset_path"], allow_pickle=True)

    labels = (np.array(X[:, 0])).astype(np.float32)
    X = np.array(X[:, 1].tolist())
    Label = labels

    X = X[:, :, 200 * 2:200 * 6]
    X = X[:, :, 0:-1:3]

    combined_indices = np.where(np.isin(Label, config["data"]["selected_classes"]))[0]
    # Use the combined indices to index X and Label
    X = X[combined_indices, :, :]
    Label = Label[combined_indices]

    c = np.random.permutation(X.shape[0])
    X = X[c, :, :]
    Label = Label[c]

    # Get unique values and create a mapping
    unique_values = np.unique(Label)
    mapping = {val: idx for idx, val in enumerate(unique_values)}

    # Apply the mapping to the Label array
    Label = np.array([mapping[val] for val in Label])

    X_train, X_test, y_train, y_test = train_test_split(X, Label, test_size=config["final"]["test_split"],
                                                        random_state=0)
    dataloader = load_data(x=X_train, y=y_train, device=device, config=config, shuffle=True)
    test_dataloader = load_data(x=X_test, y=y_test, device=device, config=config)

    model, acc = train_model(dataloader=dataloader, test_dataloader=test_dataloader, config=config, device=device)

    print(f'Final acc: {acc}')
    torch.save(model.state_dict(), config["model"]["path"])
    return