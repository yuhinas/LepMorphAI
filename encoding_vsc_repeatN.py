import torch

import os

import numpy as np
import pandas as pd

import fire

from networks import *
from utils import str_to_list, load_model, is_image_file
from dataset import *
from touch_dir import touch_dir

def encoding_vsc(epoch_num=40000, model=None):

    code_dim = 512

    model_id = 'vsc_epoch_%d' % epoch_num

    #--------------build models -------------------------
    if model is None:
        model = VSC(cdim=3, hdim=code_dim, channels=str_to_list('32, 64, 128, 256, 512, 512'), image_size=256).cuda()
        load_model(model, './pretrained/%s.pth' % model_id)

    model.eval()
    #print(model)
    
    dataroot = "./worldwide_lepidoptera_yolov4_cropped_and_padded_20210907"

    #-----------------load dataset--------------------------
    image_list = [dataroot + '/' + x for x in os.listdir(dataroot) if is_image_file(x)]  
    train_list = image_list[:len(image_list)]
    #train_list = image_list[:38]
    assert len(train_list) > 0
    print (len(train_list))
    
    train_set = ImageDatasetFromFile(train_list, aug=False)
    batch_size = 50
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

    mu_s = np.empty((len(train_list), code_dim))
    logvar_s = np.empty((len(train_list), code_dim))
    logspike_s = np.empty((len(train_list), code_dim))

    all_filenames = np.array([])
    with torch.no_grad():
        for iteration, (batch, _, filenames) in enumerate(train_data_loader, 0):
            
            print(iteration, end='\r')
            #print(filenames)
            
            real= Variable(batch).cuda()

            mu, logvar, logspike = model.encode(real)
            #z = model.reparameterize(mu, logvar, logspike)

            all_filenames = np.append(all_filenames, filenames)

            from_ = iteration * batch_size
            to_ = from_ + batch.size(0)

            mu_s[from_:to_,] = mu.detach().data.cpu().numpy()
            logvar_s[from_:to_,] = logvar.detach().data.cpu().numpy()
            logspike_s[from_:to_,] = logspike.detach().data.cpu().numpy()

            del real
            del mu
            del logvar
            del logspike
        print()

    with torch.no_grad():
        repeatN = 1000
        codes_ = model.reparameterize(torch.from_numpy(mu_s).cuda(), torch.from_numpy(logvar_s).cuda(), torch.from_numpy(logspike_s).cuda())
        for rep_ in range(1, repeatN):
            print('Repeating %d' % rep_, end='\r')
            codes_ = codes_ + model.reparameterize(torch.from_numpy(mu_s).cuda(), torch.from_numpy(logvar_s).cuda(), torch.from_numpy(logspike_s).cuda())
        print()
        codes = (codes_ / repeatN).detach().data.cpu().numpy()

    df = pd.DataFrame(data=codes, columns=range(codes.shape[1]))
    mu_df = pd.DataFrame(data=mu_s, columns=range(mu_s.shape[1]))
    logvar_df = pd.DataFrame(data=logvar_s, columns=range(logvar_s.shape[1]))
    logspike_df = pd.DataFrame(data=logspike_s, columns=range(logspike_s.shape[1]))
    
    df['filename'] = all_filenames
    mu_df['filename'] = all_filenames
    logvar_df['filename'] = all_filenames
    logspike_df['filename'] = all_filenames
    #print(df)

    to_save = "./save/%s" % model_id
    touch_dir(to_save)

    #np.save("%s/codes.npy" % to_save, codes)
    df.to_csv("%s/codes.csv" % to_save, sep="\t")
    mu_df.to_csv("%s/mu_s.csv" % to_save, sep="\t")
    logvar_df.to_csv("%s/logvar_s.csv" % to_save, sep="\t")
    logspike_df.to_csv("%s/logspike_s.csv" % to_save, sep="\t")

    # read from codes.csv to check file status, or just copy df to var codes
    # codes = df.copy()
    codes = pd.read_csv(f'save/vsc_epoch_{epoch_num}/codes.csv', sep='\t', index_col=0)
    families = [f.split('/')[-1].split('_')[0] for f in codes.filename.values]
    subfamilies = [f.split('/')[-1].split('_')[0] + '_' + f.split('/')[-1].split('_')[1] for f in codes.filename.values]
    sps = ['_'.join(f.split('/')[-1].split('_')[2:4]) for f in codes.filename.values]

    # subfamilies[:10]
    # np.unique(families).shape

    codes['family'] = families
    codes['subfamily'] = subfamilies
    codes['sp'] = sps
    codes.to_csv(f'save/vsc_epoch_{epoch_num}/sp_all.csv', sep='\t', index=False)



if __name__ == '__main__':
    fire.Fire(encoding_vsc)
