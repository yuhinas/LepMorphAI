from numpy.core.numeric import isclose
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
import numpy as np
from touch_dir import touch_dir
from codes_utils import codes_to_no_neg, codes_to_with_neg
from sklearn.model_selection import train_test_split
from average_meter import AverageMeter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

class Tools(nn.Module):

    def __init__(self):
        super(Tools, self).__init__()
        self.c = 50

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        #selection = F.sigmoid(125 * (eta + logspike.exp() - 1))
        selection = F.sigmoid(self.c * (eta + logspike.exp() - 1))
        return selection.mul(gaussian)


class GeneralDataset(data.Dataset):
    def __init__(self, x, y):
        super(GeneralDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class FamilyPrediction(nn.Module):

    def __init__(self, in_ch=1024, n_classes=97, n_neurons=512, n_layers=2, shrink_rate=2):
        super(FamilyPrediction, self).__init__()

        negative_slope = 0.2

        predict_family = [
            #nn.BatchNorm1d(512),
            nn.Linear(in_ch, n_neurons),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
        ]

        for _ in range(n_layers):
            predict_family.extend([
                nn.Dropout(0.5),
                #nn.BatchNorm1d(n_neurons),
                nn.Linear(n_neurons, int(n_neurons//shrink_rate)),
                #nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
            ])
            n_neurons //= shrink_rate
            n_neurons = int(n_neurons)

        predict_family.extend([
            #nn.BatchNorm1d(n_neurons),
            nn.Linear(n_neurons, n_classes),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
        ])

        self.predict_family =nn.Sequential(*predict_family)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        pred = self.predict_family(x)
        return pred

epoch_num = 40000
model_id = 'vsc_epoch_%s' % epoch_num   # epoch_5640
subdir = model_id + '/pred_fam'    # vsc/epoch_5640/pred_fam
profile_dir = subdir + '/' + 'no_neg'
model_path = "./save/%s/subfam_pred_with_vsc_reparams.pkl" % (profile_dir)

########################
codes_ = pd.read_csv('save/vsc_epoch_%d/codes.csv' % epoch_num, sep='\t', index_col=0)

mu_s = pd.read_csv('save/vsc_epoch_%d/mu_s.csv' % epoch_num, sep='\t', index_col=0)
logvar_s = pd.read_csv('save/vsc_epoch_%d/logvar_s.csv' % epoch_num, sep='\t', index_col=0)
logspike_s = pd.read_csv('save/vsc_epoch_%d/logspike_s.csv' % epoch_num, sep='\t', index_col=0)

codes = pd.concat([mu_s.iloc[:,:512], logvar_s.iloc[:,:512], logspike_s.iloc[:,:512], codes_], axis=1)

families = [f.split('/')[-1].split('_')[0] for f in codes.filename.values]
subfamilies = [f.split('/')[-1].split('_')[0] + '_' + f.split('/')[-1].split('_')[1] for f in codes.filename.values]
sps = ['_'.join(f.split('/')[-1].split('_')[2:4]) for f in codes.filename.values]

subfamilies[:10]
sps[:10]

np.unique(families).shape
np.unique(subfamilies).shape

codes['family'] = families
codes['subfamily'] = subfamilies
codes['sp'] = sps

df_ = codes.copy()

########################

# df_ = pd.read_csv(f"./save/{subdir}/../sp_all.csv", sep="\t")
# df_

#df_.groupby('sp').head(1).groupby('subfamily').size().quantile(.5)
#df_.subfamily[~df_.subfamily.duplicated()].to_csv('fam_subfam.csv', index=False)

np.unique(df_.family, return_inverse=True)

#subfams = np.load('worldwide_family_subfamily_arr_20210610.npy', allow_pickle=True)
#subfams.shape

subfams, subfam_ids = np.unique(df_.subfamily, return_inverse=True)
#np.save('worldwide_family_subfamily_arr_20210908.npy', subfams)

n_classes = len(subfams)
model = FamilyPrediction(in_ch=1024, n_neurons=256, n_layers=1, shrink_rate=1, n_classes=n_classes).cuda()

#subfam_idv_counts_threshold = 0

touch_dir(f'./save/{profile_dir}')

print(n_classes)
df_['subfam_id'] = subfam_ids

subfam_sp_size = df_.groupby(['subfamily', 'sp']).head(1).groupby('subfamily').size().to_frame('num_of_sp').reset_index()
subfam_sp_size_gteN = subfam_sp_size[subfam_sp_size.num_of_sp >= 0].subfamily.values
# fam_sp_size.num_of_sp >= 1 # Test Acc: 0.884241
# fam_sp_size.num_of_sp >= 3 # Test Acc: 0.889079
df = df_[df_.subfamily.isin(subfam_sp_size_gteN)]
# fam_idv_counts = df.groupby('family').size()
# df = df[df.family.isin(fam_idv_counts[fam_idv_counts >= fam_idv_counts_threshold].index.values)]
df = df.reset_index(drop=True)

n_cases = len(df)

# codes512 = df.iloc[:,1024:1536].values
# codes1024 = codes_to_no_neg(codes512)

x = df.iloc[:,:2048].values
y = df.subfam_id.values

#x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=.7, test_size=.3, random_state=42, stratify=y)
x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=.7, test_size=.3, random_state=42)

batch_size = 100

train_set = GeneralDataset(x_train, y_train)
train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = GeneralDataset(x_test, y_test)
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.1, 0.999), weight_decay=2e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=2e-4)


###########################################

min_valid_loss = np.inf
max_valid_acc = -np.inf

n_epoch = 10000
earlystop_num = 100
earlystop_cnt = 0
# epoch = 0
tools = Tools()

for epoch in range(n_epoch):

    # Training
    model.train()

    train_loss = AverageMeter()

    for train_batch_idx, (x_, y) in enumerate(train_data_loader, 0):

        mu_ = x_[:,:512]
        logvar_ = x_[:,512:1024]
        logspike_ = x_[:,1024:1536]

        x_train_batch_cuda_ = tools.reparameterize(mu_.cuda(), logvar_.cuda(), logspike_.cuda())
        x_train_batch_cuda = codes_to_no_neg(x_train_batch_cuda_, is_cuda=True)

        #x_train_batch_cuda = x_no_neg.float().cuda()
        y_train_batch_cuda = y.long().cuda()

        pred = model(x_train_batch_cuda.float())
        loss = criterion(pred, y_train_batch_cuda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item())
        #print("Epoch %04d (%03d/%03d): Training Loss: %.6f/%.6f" % (epoch+1, train_batch_idx, train_n_batches, train_loss.val, train_loss.avg), end="\r")
        print("Epoch %04d (%03d/%03d): Training Loss: %.6f/%.6f" % (epoch+1, train_batch_idx, -1, train_loss.val, train_loss.avg), end="\r")
        del x_train_batch_cuda
        del y_train_batch_cuda
        del pred

    #print()

    # Validating
    model.eval()
    valid_loss = AverageMeter()
    valid_f1 = AverageMeter()
    valid_acc = AverageMeter()
    valid_bacc = AverageMeter()
    for valid_batch_idx, (x_, y) in enumerate(test_data_loader, 0):

        x_valid_batch_cuda_ = x_[:,1536:2048].cuda()
        x_valid_batch_cuda = codes_to_no_neg(x_valid_batch_cuda_, is_cuda=True)

        #x_valid_batch_cuda = x_no_neg.float().cuda()
        y_valid_batch_cuda = y.long().cuda()

        pred = model(x_valid_batch_cuda.float())
        loss = criterion(pred, y_valid_batch_cuda)

        pred_flatten = np.argmax(pred.data.detach().cpu().numpy(), axis=1)

        f1 = f1_score(y, pred_flatten, average='weighted')
        valid_f1.update(f1)
        acc = accuracy_score(y, pred_flatten)
        bacc = balanced_accuracy_score(y, pred_flatten)
        valid_acc.update(acc)
        valid_bacc.update(bacc)

        valid_loss.update(loss.item())
        print("Epoch %04d (%03d/%03d): Validation Loss: %.6f/%.6f" % (epoch+1, valid_batch_idx, -1, valid_loss.val, valid_loss.avg), end="\r")
        del x_valid_batch_cuda
        del y_valid_batch_cuda
        del pred

    touch_dir("./save/%s" % profile_dir)

    with open('./save/%s/losses.log' % (subdir),'a') as loss_log:
        loss_log.write(
            "\t".join([
                str('no_neg'),
                str(epoch+1), 
                str(train_loss.avg), 
                str(valid_loss.avg),
                str(valid_f1.avg),
                str(valid_acc.avg),
                str(valid_bacc.avg) + "\n"
            ])
        )

    #if valid_loss.avg < min_valid_loss:
    if valid_acc.avg > max_valid_acc:
        earlystop_cnt = 0
        #min_valid_loss = valid_loss.avg
        max_valid_acc = valid_acc.avg
        according_valid_f1 = valid_f1.avg
        according_valid_acc = valid_acc.avg
        according_valid_bacc = valid_bacc.avg
        according_epoch = epoch + 1
        torch.save(model.state_dict(), model_path)
        #print ("\nSave new minimum %.6f" % min_valid_loss)
        print ("\nSave new maximum %.6f" % max_valid_acc)
        print("Weight F1 score: %.6f" % valid_f1.avg)
        print("Accuracy score : %.6f" % valid_acc.avg)
        print("Balanced accuracy score : %.6f" % valid_bacc.avg)
        print('no_neg')
    else:
        earlystop_cnt += 1
        pass
        #print()
    if earlystop_cnt > earlystop_num:
        print("Early stop!")
        break
