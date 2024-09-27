# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:31:21 2023

@author: Shayan
"""

import scipy.io
import torch
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
from LMDA_modified import LMDA

filename = 'D:/Mirage91/npi/features/cleaned_epoched_eeg.npy'
X = np.load(filename, allow_pickle=True)


labels = (np.array(X[:, 0])).astype(np.float32)  
inputs = np.array(X[:, 1].tolist())  
X = inputs
Label = labels
# Label[Label==1] = 0
# Label[Label==2] = 1
# Label[Label==3] = 2
# Label[Label==4] = 3

'cross validation'
X = X[:,:,200*2:200*6]
X = X[:,:,0:-1:3]


c1 = np.where(Label == 3)[0]  # Get indices where Label is 3
c2 = np.where(Label == 4)[0]  # Get indices where Label is 4
c3 = np.where(Label == 1)[0]  # Get indices where Label is 2



# Combine the indices
combined_indices = np.concatenate((c1, c2,c3))

# Use the combined indices to index X and Label
X = X[combined_indices, :, :]
Label = Label[combined_indices]

c = np.random.permutation(X.shape[0])
X = X[c,:,:]
Label = Label[c]


# Label[c1] = 0
# Label[c2] = 1
# Label[c3] = 2
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
criterion_cls = torch.nn.CrossEntropyLoss().cuda()

# model = Conformer().cuda()
# model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
# model = model.cuda()

# model = EEGInception().cuda()

'mine'
out_channels = 15
n_chunks = 3
depth = 15

chans = 63
samples = 121
num_classes = 2
# 

C = 63
T = 121
# model = LMDA(num_classes=2, chans=32, samples=438, depth=9, kernel=75, channel_depth1=24, channel_depth2=9).cuda()

# model =  DeepConvNet(C, T,  5, 0.5, N = 2, pool_type = 'maxpool', max_norm_ratio = 1).cuda()
    
# model = EEGInception(input_time=1890, fs=64, ncha=chans, filters_per_branch=8,
#                   scales_time=(500, 250, 125), dropout_rate=0.25,
#                   activation=nn.ELU(inplace=True), n_classes=2).cuda()
# model = ShallowConvNet(num_classes=2, chans=chans, samples=samples).cuda()

    # model = EEGNet(num_classes = 2, chans = 63, samples=128, dropout_rate=0.5, kernel_length=32, F1=8, F2=16).cuda()

# n_classes, dropoutRate, kernelLength, kernelLength2, F1, D = 4, 0.5, 64, 16, 8, 2
# F2 = F1 * D
# model = EEGNet(n_classes, chans, samples, dropoutRate, kernelLength, kernelLength2, F1, D, F2).cuda()

# model = YourModel(20, 20, 20,(2,2,2), (2,4,4), (2,2,2)).cuda()


# kernel_size = [(7, 3, 3), (5, 3, 3),(3, 3, 3),(3, 3)]
# chan = (16,16,16,16)
# embed_in = 32
# hidden_dimension = 32
# num_heads = 4
# head_dim = 8
# window_size = 3
# relative_pos_embedding=True
# downscaling_factor=4
# # x = torch.rand(1,1,64,32,32)
# # _, _, depth, _, _ = x.shape
# model = SwinCNN(64,kernel_size, chan, embed_in, hidden_dimension,num_heads,head_dim,window_size,downscaling_factor,relative_pos_embedding).cuda()

# kernel_size = [(7, 3, 3), (5, 3, 3),(3, 3, 3),(1, 1)]
# chan = (16,16,16,16)

# num_heads = 5
# head_dim = 8
# relative_pos_embedding = True
# downscaling_factor = 2
# embed_in=40
# hidden_dimension = 40
# window_size = 2 


# kernel_size = [(7, 3, 3), (5, 3, 3),(3, 3, 3),(1, 1)]
# chan = (16,16,16,16)

# num_heads = 5
# head_dim = 8
# relative_pos_embedding = True
# downscaling_factor = 2
# embed_in=40
# hidden_dimension = 40
# window_size = 2 
# x = torch.rand(1,1,64,32,32)
# _, _, depth, _, _ = x.shape
# model = SwinCNN4(depth,kernel_size, chan, embed_in, hidden_dimension,num_heads,head_dim,window_size,downscaling_factor,relative_pos_embedding).cuda()


########################################

# model = EEGNet(num_classes = 4, chans = 32, samples=875, dropout_rate=0.5, kernel_length=32, F1=8, F2=16).cuda()

# a = torch.randn(12, 1, 63, 128).cuda().float()
# l2 = model(a)
# model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# model = ShallowConvNet(num_classes=2, chans=63, samples=128).cuda()
# model = EEGNet(num_classes = 2, chans = 63, samples=128, dropout_rate=0.5, kernel_length=32, F1=8, F2=16).cuda()

# model = ConvTransformer(num_classes=2, channels=8, num_heads=2, E=16, F=256, T=64, depth=2).cuda()
# model = ConvTransformer(num_classes=2, channels=8, num_heads=2, E=16, F=256, T=128, depth=2).cuda()


# model = TimeSformer(img_size=32, num_classes=2, num_frames=64, attention_type='divided_space_time').cuda()

# x = torch.rand((128,63,6000))
# label = torch.randint(low=0, high=2, size=(6000,))

from sklearn.model_selection import KFold
import numpy as np
kf = KFold(n_splits=10)
Acc_all = np.zeros((10,250))

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

    img = (torch.from_numpy(train_data)).type(Tensor)
    label = (torch.from_numpy(y_train)).type(LongTensor)
    
    # label = torch.transpose(label, 0, 1)
    
    dataset = torch.utils.data.TensorDataset(img, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size= batch_sizee, shuffle=True)
    
    test_data = np.expand_dims(X_test, axis=1)
    test_data = (torch.from_numpy(test_data)).type(Tensor)
    test_label = (torch.from_numpy(y_test)).type(LongTensor)

    # test_data = (torch.from_numpy(test_data)).type(torch.float32)
    # test_label = (torch.from_numpy(test_label)).type(torch.float32)

    # test_label = torch.from_numpy(test_label - 1)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_label)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_sizee, shuffle=False)
    
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size= batch_sizee, shuffle=True,pin_memory=True)
    del img, label, train_data
    torch.cuda.empty_cache()
    model = LMDA(num_classes=3, chans=32, samples=267, channel_depth1=24, channel_depth2=7).cuda()
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
            train_acc.append(float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0)))
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
                Acc_all[fold-1,e] = acc
         
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

    

    
    
    
acc = np.mean(Acc_all,axis = 0)


'runing time'
# import time
# x = torch.rand(1, 1, 63, 121)
# x = x.cuda()

# timin = np.zeros((100))

# for i in range(100):
#     start_time = time.time()
#     predictions = model(x )
#     end_time = time.time()
#     time_taken = end_time - start_time
#     timin[i] = time_taken
    

# np.mean(timin)*100


'number of parameters'
# import numpy as np

# def count_parameters(model):
#     total_params = 0
#     for param in model.parameters():
#         total_params += np.prod(param.size())
#     return total_params

# # Example usage:
# # Assuming you have a deep learning model named 'model'
# # (e.g., a PyTorch or TensorFlow model)

# num_params = count_parameters(model)

# params = list(model.parameters())



'Results'

'EEGNET'
# class 1 & 2 -> 67-70
# class 1 & 3 -> 53-54
# class 1 & 4 -> 70-72

# class 2 & 3 -> 68-71
# class 2 & 4 -> 72-74

# class 3 & 4 -> 75-78


'LMDA'
# class 3 & 4 -> 78-81

# class 3,4,2  -> 50-52
# class 1,4,2  -> 48-50
# class 1,2,3  -> 51-52



# class 3,4,1  -> 64-67



