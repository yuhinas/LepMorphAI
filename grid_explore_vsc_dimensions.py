import torch

#torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
from torchvision.utils import save_image, make_grid

from networks import VSC
from utils import str_to_list, load_model, is_image_file

from touch_dir import touch_dir

import os

from skimage import io as io
from PIL import Image

from PIL import ImageFont
from PIL import ImageDraw

import numpy as np

import fire

from dim_hist_sampling import dim_hist_sampling

def grid_explore_dimensions(epoch_num=40000, model=None):

    code_dim = 512
    model_id = f'vsc_epoch_{epoch_num}'

    #--------------build models -------------------------
    if model is None:
        model = VSC(cdim=3, hdim=code_dim, channels=str_to_list("32, 64, 128, 256, 512, 512"), image_size=256).cuda()
        load_model(model, f'./pretrained/{model_id}.pth')

    model.eval()

    to_save_path = './save/%s/feature_viz_simplified' % model_id
    to_gif_save_path = './save/%s/feature_viz_simplified/gif' % model_id
    to_diff_save_path = './save/%s/feature_viz_simplified/diff' % model_id
    touch_dir(to_save_path)
    touch_dir(to_gif_save_path)
    touch_dir(to_diff_save_path)


    '''
    load pre_trained_model
    '''

    # selected dim, range 0 ~ code_dim
    selected_dim = range(code_dim)
    
    # num of x tic
    start = -3
    step = 0.6
    end = -start + step
    
    x_tic = np.arange(start, end, step, dtype=float)
    try:
        q005 = np.load("./save/%s/q005.npy" % model_id)
    except:
        dim_hist_sampling(epoch_num=epoch_num, model=model, is_vsc=True)
        q005 = np.load("./save/%s/q005.npy" % model_id)


    # template

    selected = 0
    for selected in selected_dim:

        x_tic = q005[selected]
        x_tic = x_tic[[0, 6, 12, 18, 27, 35, 41, 47, 53]]
        # codes dim: N x code_dim
        N = len(x_tic)
        # reverse_idx = np.arange(int(N/2)-1, -1, -1)

        codes = np.zeros((N, code_dim))
        for i in range(N):
            codes[i, selected] = x_tic[i]

        codes = torch.from_numpy(codes).float().cuda()

        images = model.decode(codes)
        
        # images_1st_half = images[reverse_idx]
        # images_2nd_half = images[27:]

        save_image(images, os.path.join(to_save_path, "feature_%04d.jpg" % selected), nrow=9)
        # change to True for generate more forms
        if False:
            touch_dir(os.path.join(to_save_path, 'no_neg'))
            save_image(images_1st_half, os.path.join(to_save_path, 'no_neg', "feature_%04d.jpg" % (selected + 512)), nrow=9)
            save_image(images_2nd_half, os.path.join(to_save_path, 'no_neg', "feature_%04d.jpg" % (selected)), nrow=9)

            diff_pos = images_1st_half[-1] - images_1st_half[0]
            diff_neg = -diff_pos.clone()
            diff_pos[diff_pos < 0] = 0
            diff_neg[diff_neg < 0] = 0
            touch_dir(os.path.join(to_save_path, 'no_neg/diff'))
            save_image([diff_pos, diff_neg], os.path.join(to_save_path, 'no_neg/diff', "feature_%04d.jpg" % (selected + 512)), nrow=2)
            del diff_pos, diff_neg
            
            diff_pos = images_2nd_half[-1] - images_2nd_half[0]
            diff_neg = -diff_pos.clone()
            diff_pos[diff_pos < 0] = 0
            diff_neg[diff_neg < 0] = 0
            touch_dir(os.path.join(to_save_path, 'no_neg/diff'))
            save_image([diff_pos, diff_neg], os.path.join(to_save_path, 'no_neg/diff', "feature_%04d.jpg" % (selected)), nrow=2)
            del diff_pos, diff_neg

            images_npy = images.detach().permute(0,2,3,1).data.cpu().numpy()
            imgs_to_append = []

            font = ImageFont.truetype('arial.ttf', 20)

            for b in range(1, images_npy.shape[0]):
                im = Image.fromarray(np.uint8(images_npy[b] * 255))
                draw = ImageDraw.Draw(im)
                draw.text((5, 5), str(b), (0, 0, 255), font=font)

                imgs_to_append.append(im)

            img = Image.fromarray(np.uint8(images_npy[0] * 255))
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), str(0), (0, 0, 255), font=font)
            img.save(fp=os.path.join(to_gif_save_path, "feature_%04d.gif" % selected), format='GIF', append_images=imgs_to_append,
                save_all=True, duration=200, loop=0)

            im_last_npy = images_npy[-1]
            im_first_npy = images_npy[0]
            dim_diff = im_last_npy - im_first_npy
            dim_rev_diff = -dim_diff
            dim_diff[dim_diff < 0] = 0
            dim_rev_diff[dim_rev_diff < 0] = 0

            dim_diff = np.uint8(dim_diff * 255)
            dim_rev_diff = np.uint8(dim_rev_diff * 255)

            dim_diff_set = np.concatenate([dim_diff, dim_rev_diff], axis=1)

            diff_img = Image.fromarray(dim_diff_set)
            diff_img.save(fp=os.path.join(to_diff_save_path, "dim_diff_%04d.jpg" % selected), format='JPEG')

        try:
            del images
            del codes
            del images_npy
            del images_1st_half, images_2nd_half
        except:
            pass

if __name__ == '__main__':
    fire.Fire(grid_explore_dimensions)
