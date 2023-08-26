# python main_vsc.py 
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import VSC
#from model import AAResNet50NoTop
from resnet50_classifier import ResNet50_family_classifier_notop
from math import log10
import torchvision
from torchvision.utils import make_grid, save_image

from average_meter import AverageMeter
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_model(model, optims, pretrained):
    weights = torch.load(pretrained)
    
    try:
        pretrained_model_dict = weights['model'].state_dict()
    except:
        pretrained_model_dict = weights
    model_dict = model.state_dict()
    pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_model_dict)
    model.load_state_dict(model_dict)
    
    if len(optims) > 0:
        pretrained_optims = weights['optims']
        assert(isinstance(pretrained_optims, list))
        for optim_idx, pretrained_optim in enumerate(pretrained_optims):
            pretrained_optim_dict = pretrained_optim.state_dict()
            optim_dict = optims[optim_idx].state_dict()
            pretrained_optim_dict = {k: v for k, v in pretrained_optim_dict.items() if k in optim_dict}
            optim_dict.update(pretrained_optim_dict)
            optims[optim_idx].load_state_dict(optim_dict)

def save_checkpoint(model, optims, epoch):
    model_out_path = "pretrained/" + "vsc_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model, "optims": optims}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])
    
def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=opt.nrow), cur_iter)
    

parser = argparse.ArgumentParser()
parser.add_argument('--channels', default="32, 64, 128, 256, 512, 512", type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=512, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=2000, help="Default=1000")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=6)

parser.add_argument('--dataroot', default="./wolrdwide_lepidoptera_yolov4_cropped_and_padded_20210407", type=str, help='path to dataset')

parser.add_argument('--trainsize', type=int, help='number of training data', default=-1)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--input_height', type=int, default=256, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=256, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=256, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=256, help='the width  of the output image to network')
parser.add_argument("--nEpochs", type=int, default=50000, help="number of epochs to train for")
parser.add_argument("--num_vsc", type=int, default=0, help="number of epochs to for vsc training")
parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--parallel', action='store_true', help='for multiple GPUs')
parser.add_argument('--outf', default='results/dfc/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument("--pretrained", default='../vsc_wwlep_glance/model/vsc/model_local_epoch_3000_iter_0.pth', type=str, help="path to pretrained model (default: none)")
parser.add_argument("--pretrained", default='', type=str, help="path to pretrained model (default: none)")

def main():

    global opt, vsc_model, resnet50
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    try:
        os.makedirs(opt.outf.strip('/') + '_valid/')
    except OSError:
        pass

    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    #if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    #if torch.cuda.is_available() and not opt.cuda:
    #    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    is_scale_back = False
    
    #--------------build VSC models -------------------------
    if opt.parallel:
        vsc_model = VSC(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height, parallel=True).cuda()
    else:
        print(opt.hdim, str_to_list(opt.channels), opt.output_height)
        vsc_model = VSC(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height).cuda()

    pretrained_default = 'pretrained/vsc_epoch_%d.pth' % opt.start_epoch
    #pretrained_default = '../vsc_wwlep_glance/model/vsc/model_local_epoch_2000_iter_0.pth'

    vsc_model.train()

    use_adam = True
    if use_adam:
        optimizerE = optim.Adam(vsc_model.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#         optimizerE = optim.AdamW(vsc_model.encoder.parameters(), lr=opt.lr)
        optimizerG = optim.Adam(vsc_model.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#         optimizerG = optim.AdamW(vsc_model.encoder.parameters(), lr=opt.lr)
        optimizers = [optimizerE, optimizerG]
    else:
        optimizerE = optim.RMSprop(vsc_model.encoder.parameters(), lr=opt.lr)
        optimizerG = optim.RMSprop(vsc_model.decoder.parameters(), lr=opt.lr)
        optimizers = []
    
    if os.path.isfile(pretrained_default):
        if opt.start_epoch > 0:
            print ("Loading default pretrained %d..." % opt.start_epoch)
            load_model(vsc_model, optimizers, pretrained_default)
        else:
            print ("Start training from scratch...")
    elif opt.pretrained:
        print ("Loading pretrained from %s..." % opt.pretrained)
        load_model(vsc_model, optimizers, opt.pretrained)

    
    #--------------build ResNet50 models -------------------------
    resnet50 = ResNet50_family_classifier_notop()
    #load_model(resnet50, [], '/home/ess/AI_projects/gsmai/aa_resnet50/model_saved/model_local_epoch_636.pth')
    load_model(resnet50, [], './pretrained/pretrained_fam_classification_resnet50_20210613.pth')
    resnet50 = nn.DataParallel(resnet50)
    resnet50.cuda()
    resnet50.eval()
    
    #-----------------load dataset--------------------------
    image_list = [x for x in glob.iglob(opt.dataroot + '/**/*', recursive=True) if is_image_file(x)]
    #train_list = image_list[:opt.trainsize]
    # train_list = image_list[:]
    # assert len(train_list) > 0
    
    image_cache = np.load('./wolrdwide_lepidoptera_yolov4_cropped_and_padded.npy', allow_pickle=True)
    
    # train_set = ImageDatasetFromFile(train_list, aug=True)
    train_set = ImageDatasetFromCache(image_cache, aug=True)
    
    #dfc_data_loader = torch.utils.data.DataLoader(train_set, batch_size=(opt.batchSize // 8) * 3, shuffle=True, num_workers=int(opt.workers), drop_last=True, pin_memory=True)
    dfc_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True, pin_memory=True)
    vsc_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True, pin_memory=True)
    
    valid_list = ['./benchmarks/' + x for x in os.listdir('./benchmarks') if is_image_file(x)]
    valid_set = ImageDatasetFromFile(valid_list, aug=False)
    valid_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    blur = torchvision.transforms.GaussianBlur(5)
    start_time = time.time()

    #cur_iter = 0        
    #cur_iter = int(np.ceil(float(opt.trainsize) / float(opt.batchSize)) * opt.start_epoch)
#     cur_iter = len(train_data_loader) * (opt.start_epoch - 1)
#     if opt.test_iter > len(train_data_loader) - 1:
#         opt.test_iter = len(train_data_loader) - 1
    
    #----------------Train func
    def train_dfc(epoch, iteration, batch, denoised_batch, cur_iter, dfc=True):

        batch_size = batch.size(0)

        real = Variable(batch).cuda() 
        denoised = Variable(denoised_batch).cuda()

        noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda()
        fake = vsc_model.sample(noise)

        time_cost = time.time()-start_time
        info = "====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(data_loader), time_cost)

        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'

        #=========== Calc Losses and Update Optimizer ================                  
        
        real_mu, real_logvar, real_logspike, z, rec = vsc_model(real)
        
        loss_rec =  vsc_model.reconstruction_loss(rec, denoised, True)
        #loss_rec =  0

        loss_prior = vsc_model.prior_loss(real_mu, real_logvar, real_logspike)

        if dfc:
            *denoised_feature_levels, _ = resnet50(blur(denoised.cuda()))
            *rec_feature_levels, _ = resnet50(blur(rec.cuda()))

#             print(denoised_feature_levels[0].size())
#             print(denoised_feature_levels[1].size())
#             print(denoised_feature_levels[2].size())
#             print(denoised_feature_levels[3].size())
            
            levelN = 3
#             loss_feature_levelN = \
#                 ((denoised_feature_levels[levelN-3].reshape(batch_size, 2**(levelN-0+2), -1) - rec_feature_levels[levelN-3].reshape(batch_size, 2**(levelN-0+2), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1. + \
#                 ((denoised_feature_levels[levelN-2].reshape(batch_size, 2**(levelN-1+2), -1) - rec_feature_levels[levelN-2].reshape(batch_size, 2**(levelN-1+2), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1. + \
#                 ((denoised_feature_levels[levelN-1].reshape(batch_size, 2**(levelN-2+2), -1) - rec_feature_levels[levelN-1].reshape(batch_size, 2**(levelN-2+2), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1. + \
#                 ((denoised_feature_levels[levelN-0].reshape(batch_size, 2**(levelN-3+2), -1) - rec_feature_levels[levelN-0].reshape(batch_size, 2**(levelN-3+2), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1.

            loss_feature_levelN = \
                ((denoised_feature_levels[levelN-3].reshape(batch_size, 2**(levelN+5), -1) - rec_feature_levels[levelN-3].reshape(batch_size, 2**(levelN+5), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1. + \
                ((denoised_feature_levels[levelN-2].reshape(batch_size, 2**(levelN+6), -1) - rec_feature_levels[levelN-2].reshape(batch_size, 2**(levelN+6), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1. + \
                ((denoised_feature_levels[levelN-1].reshape(batch_size, 2**(levelN+7), -1) - rec_feature_levels[levelN-1].reshape(batch_size, 2**(levelN+7), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1. + \
                ((denoised_feature_levels[levelN-0].reshape(batch_size, 2**(levelN+8), -1) - rec_feature_levels[levelN-0].reshape(batch_size, 2**(levelN+8), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 1.

            loss_feature_levelN /= 100
#             loss_feature_levelN = \
#                 ((denoised_feature_levels[levelN].reshape(batch_size, 2**(levelN+2), -1) - rec_feature_levels[levelN].reshape(batch_size, 2**(levelN+2), -1)) ** 2).sum(dim=2).sum(dim=1).mean() / 20.
        
        
            loss = loss_rec.cuda() + 2 * loss_prior.cuda() + loss_feature_levelN.cuda()
        else:
            loss = loss_rec + loss_prior

        optimizerG.zero_grad()
        optimizerE.zero_grad()
        loss.backward()                   
        optimizerE.step()
        optimizerG.step()

        try:
            am_rec.update(loss_rec.item())
        except:
            am_rec.update(0)
        
        am_prior.update(loss_prior.item())
        
        try:
            am_levelN.update(loss_feature_levelN.item())
        except:
            am_levelN.update(0)

        info += 'Rec: {:.4f}({:.4f}), Prior: {:.4f}({:.4f}), LvN: {:.4f}({:.4f}), '.format(am_rec.val, am_rec.avg, am_prior.val, am_prior.avg, am_levelN.val, am_levelN.avg)
        print(info, end='\r')

        if cur_iter % opt.test_iter is 0:  
            vutils.save_image(torch.cat([real[:(opt.nrow*2)], rec[:(opt.nrow*2)], fake[:(opt.nrow*2)]], dim=0).data.cpu(), '{}/image_e{}i{}.jpg'.format(opt.outf, epoch, cur_iter),nrow=opt.nrow)
            vutils.save_image(torchvision.transforms.Resize(128)(torch.cat([real[:(opt.nrow*2)], rec[:(opt.nrow*2)], fake[:(opt.nrow*2)]], dim=0)).data.cpu(), '{}/image128_e{}i{}.jpg'.format(opt.outf, epoch, cur_iter),nrow=opt.nrow)
            with open('./results/dfc_losses.log','a') as loss_log:
                loss_log.write(
                    "\t".join([
                        str(epoch),
                        str(cur_iter),
                        '%.4f' % am_rec.avg,
                        '%.4f' % am_prior.avg,
                        '%.4f\n' % am_levelN.avg
                    ])
                )
        elif cur_iter % 100 is 0:
            vutils.save_image(torch.cat([real[:(opt.nrow*2)], rec[:(opt.nrow*2)], fake[:(opt.nrow*2)]], dim=0).data.cpu(), '{}/image_up_to_date.jpg'.format(opt.outf),nrow=opt.nrow)
            vutils.save_image(torchvision.transforms.Resize(128)(torch.cat([real[:(opt.nrow*2)], rec[:(opt.nrow*2)], fake[:(opt.nrow*2)]], dim=0)).data.cpu(), '{}/image128_up_to_date.jpg'.format(opt.outf),nrow=opt.nrow)
        elif (cur_iter + 1) % len(data_loader) is 0:  
            with open('./results/dfc_losses.log','a') as loss_log:
                loss_log.write(
                    "\t".join([
                        str(epoch),
                        str(cur_iter),
                        '%.4f' % am_rec.avg,
                        '%.4f' % am_prior.avg,
                        '%.4f\n' % am_levelN.avg
                    ])
                )
            
    #----------------Train func
    def valid_vsc(epoch, iteration, batch, denoised_batch, cur_iter):

        with torch.no_grad():
            real = Variable(batch).cuda() 
            real_mu, real_logvar, real_logspike, z, rec = vsc_model(real)
            vutils.save_image(torch.cat([real[:30], rec[:30]], dim=0).data.cpu(), './results/dfc_valid/image_epoch_{}.jpg'.format(epoch),nrow=10)
            vutils.save_image(torchvision.transforms.Resize(128)(torch.cat([real[:30], rec[:30]], dim=0)).data.cpu(), './results/dfc_valid/image128_epoch_{}.jpg'.format(epoch),nrow=10)

    
    #----------------Train by epochs--------------------------
    prev_checkpoint = None
    current_checkpoint = None
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        #save models
        save_epoch = (epoch//opt.save_iter)*opt.save_iter

        if epoch == save_epoch:
            current_checkpoint = save_checkpoint(vsc_model, optimizers, save_epoch)
            
        if prev_checkpoint is not None:
            os.remove(prev_checkpoint)
        
        prev_checkpoint = current_checkpoint
        
        vsc_model.c = 50 + epoch * vsc_model.c_delta
        
        am_rec = AverageMeter()
        am_prior = AverageMeter()
        am_levelN = AverageMeter()

        cur_iter = 0
        
        dfc = (epoch >= opt.num_vsc)
        
        if dfc:
            data_loader = dfc_data_loader
        else:
            data_loader = vsc_data_loader
        
        if opt.test_iter > len(data_loader) - 1:
            opt.test_iter = len(data_loader) - 1
            
        vsc_model.train()
        for iteration, (batch, denoised_batch, filenames) in enumerate(data_loader, 0):
            #--------------train vsc------------
            train_dfc(epoch, iteration, batch, denoised_batch, cur_iter, dfc=dfc)
            cur_iter += 1

        print()
        
        vsc_model.eval()
        for iteration, (batch, denoised_batch, filenames) in enumerate(valid_data_loader, 0):
            #--------------valid------------
            print('Validating...')
            valid_vsc(epoch, iteration, batch, denoised_batch, cur_iter)


if __name__ == "__main__":
    main()    
