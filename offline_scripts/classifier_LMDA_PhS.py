import numpy as np
import torch
from sklearn.model_selection import KFold

from offline_scripts.classifier_functions.LMDA_modified import LMDA

def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters())

def evaluate(device, config: dict):
    X = np.load(config["data"]["dataset_path"], allow_pickle=True)

    labels = (np.array(X[:, 0])).astype(np.float32)
    inputs = np.array(X[:, 1].tolist())
    X = inputs
    Label = labels

    'cross validation'
    X = X[:, :, 200 * 2:200 * 6]
    X = X[:, :, 0:-1:3]

    c1 = np.where(Label == 3)[0]  # Get indices where Label is 3
    c2 = np.where(Label == 4)[0]  # Get indices where Label is 4
    c3 = np.where(Label == 1)[0]  # Get indices where Label is 2

    # Combine the indices
    combined_indices = np.concatenate((c1, c2, c3))

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

    batch_sizee = 25
    n_epochs = 200
    c_dim = 2
    lr = 0.001
    b1 = 0.5
    b2 = 0.999

    start_epoch = 0

    Tensor = torch.cuda.FloatTensor
    # Tensor = torch.cuda.torch.cuda.HalfTensor
    LongTensor = torch.cuda.LongTensor

    # criterion_l1 = torch.nn.L1Loss().cuda()
    # criterion_l2 = torch.nn.MSELoss().cuda()
    criterion_cls = torch.nn.CrossEntropyLoss().cuda(device=device)

    kf = KFold(n_splits=10)
    Acc_all = np.zeros((10, 250))

    trind = []
    tesind = []

    fold = 1
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Label[train_index], Label[test_index]

        trind.append(train_index)
        tesind.append(test_index)

        # train_data = np.transpose(X_train, (2, 0, 1))
        train_data = np.expand_dims(X_train, axis=1)
        img = (torch.from_numpy(train_data).to(device)).type(torch.FloatTensor)
        label = (torch.from_numpy(y_train).to(device)).type(torch.LongTensor)

        # label = torch.transpose(label, 0, 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_sizee, shuffle=True)

        test_data = np.expand_dims(X_test, axis=1)
        test_data = (torch.from_numpy(test_data).to(device)).type(torch.Tensor)
        test_label = (torch.from_numpy(y_test).to(device)).type(torch.LongTensor)

        # test_data = (torch.from_numpy(test_data)).type(torch.float32)
        # test_label = (torch.from_numpy(test_label)).type(torch.float32)

        # test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_sizee, shuffle=False)

        # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size= batch_sizee, shuffle=True,pin_memory=True)
        del img, label, train_data
        if device == "cuda":
            torch.cuda.empty_cache()
        model = LMDA(num_classes=3, chans=32, samples=267, channel_depth1=24, channel_depth2=7).to(device)
        t = count_parameters(model)
        # model = Conformer(emb_size=40, depth=6, n_classes=2).cuda()

        # n_classes, dropoutRate, kernelLength, kernelLength2, F1, D = 2, 0.5, 64, 16, 8, 2
        # F2 = F1 * D
        # chans = 32
        # samples = 438
        # model = EEGNet(n_classes, chans, samples, dropoutRate, kernelLength, kernelLength2, F1, D, F2).cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        del test_data, test_label
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        Acc = []
        # Train the cnn model
        total_step = len(dataloader)
        curr_lr = lr

        train_loss = torch.zeros(n_epochs)
        test_loss = torch.zeros(n_epochs)
        bestAcc = 0
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

        for e in range(n_epochs):
            # in_epoch = time.time()
            model.train()
            loss = []
            train_acc = []
            for i, (img, label) in enumerate(dataloader):
                # if (e + 1) % 50 == 0:
                # scheduler.step()
                # img = img.cuda()
                # label = label.cuda()
                # tok, outputs = model(img)
                outputs = model(img)
                # outputs = outputs[1]
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

            # out_epoch = time.time()

            # test process
            if (e + 1) % 1 == 0:
                model.eval()
                loss_test = []
                acc = []
                with torch.no_grad():
                    for j, (img_t, label_t) in enumerate(test_dataloader):
                        # img_t = img_t.cuda()
                        # label_t = label_t.cuda()
                        Cls = model(img_t)
                        # Cls = Cls[1]
                        # Tok, Cls = model(img_t)

                        loss_test.append(criterion_cls(Cls, label_t))
                        y_pred = (torch.max(Cls, 1)[1])
                        acc.append(float((y_pred == label_t).cpu().sum().numpy()) / float(label_t.size(0)))

                    loss_test = torch.tensor(loss_test, dtype=torch.float)
                    acc = torch.tensor(acc, dtype=torch.float)
                    acc = torch.mean(acc)
                    loss_test = torch.mean(loss_test)

                    train_loss[e] = loss
                    test_loss[e] = loss_test
                    Acc.append(acc)
                    Acc_all[fold - 1, e] = acc

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc,
                      '  Best accuracy is %.6f' % bestAcc)

                scheduler.step()
                # log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    # Y_true = test_label
                    # Y_pred = y_pred
        fold += 1

    acc = np.mean(Acc_all, axis=0)