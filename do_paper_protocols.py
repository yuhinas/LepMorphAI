# coding: utf-8

"""
Last modified on 2023-02-15 10:34+8
"""

# like the linux 'touch' command, if a dir does not exist, create it, else do nothing
from touch_dir import touch_dir
import numpy as np
import pandas as pd
import torch
from torch import nn

from skimage import io
from PIL import Image
from PIL import ImageFont

from grad_cam import BackPropApp
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib as mpl

# for editable text in adobe environments
mpl.rcParams['pdf.fonttype'] = 42

# main autoencoder model
from networks import VSC
# functions to double vector dimension with no negative values, and the inverse
from codes_utils import codes_to_no_neg, codes_to_with_neg

from sklearn.decomposition import PCA
import torchvision.utils as vutils
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from torchvision.utils import save_image

# for 3d plots
import mpl_toolkits.mplot3d.axes3d as p3
from adjustText import adjust_text

from Bio import Phylo
from skbio import DistanceMatrix
from skbio.tree import nj
from io import StringIO

import re

# string to list
str_to_list = lambda x: [int(xi) for xi in x.split(',')]

# get the largest N numbers in an array
def f(a,N):
    if N > 0:
        return np.argsort(a, axis=-1)[:, ::-1][:,:N]
    elif N < 0:
        return np.argsort(a, axis=-1)[:,:-N]

# load parameters from a file into a model
def load_model(model, pretrained, map_location=None, root=False):
    weights = torch.load(pretrained, map_location=map_location)
    if root:
        pretrained_dict = weights
    else:
        pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

############################################################################################################ START OF MAIN

# model trained for 40000 epochs
epoch_num = 40000

# the dimension of the output of VSC: 512
# the doubled dimenison with no negative values: 512 * 2 = 1024
code_dim = 512 * 2

pretrained_default = f'pretrained/vsc_epoch_{epoch_num}.pth'
model_id = 'vsc_epoch_%s' % epoch_num   # vsc_epoch_40000
subdir = model_id + '/pred_fam'    # vsc_epoch_40000/pred_fam
profile_name = 'no_neg'

profile_dir = subdir + '/' + profile_name   # vsc_epoch_40000/pred_fam/no_neg
touch_dir(f'./save/{profile_dir}')

# read all features of encoded images with VSC
df_ = pd.read_csv(f"./save/{subdir}/../sp_all.csv", sep="\t")
df_.sp.unique().size

# count the number of genus
genus_list = np.unique([sp.split('_')[0] for sp in df_.sp.unique()])
len(genus_list)

# align the subfamilies in the deep feature extractor (trained ResNet50 targeting subfamilies) and the VSC
subfams = np.load('worldwide_family_subfamily_arr_20210908.npy', allow_pickle=True)
subfam_ids_ = [np.where(subfams==f)[0][0] for f in df_.subfamily.values]
n_classes = len(subfams)
print(n_classes)

# add subfamily id into the dataframe
df_['subfam_id'] = subfam_ids_

# add family id into the dataframe
fams_, fam_ids_ = np.unique(df_.family.values, return_inverse=True)
df_['fam_id'] = fam_ids_
len(fams_)

# count the number of species in each subfamily
subfam_sp_size = df_.groupby(['subfamily', 'sp']).head(1).groupby('subfamily').size().to_frame('num_of_sp').reset_index()
# keep data with subfamilies (and so families) with >= 3 species
subfam_sp_size_gteN = subfam_sp_size[subfam_sp_size.num_of_sp >= 3].subfamily.values
df = df_[df_.subfamily.isin(subfam_sp_size_gteN)]
df = df.reset_index(drop=True)
n_cases = len(df)
print(n_cases)

# keep the fam_ids and subfam_ids accordingly
subfam_ids = np.array(subfam_ids_)[df_.subfamily.isin(subfam_sp_size_gteN)]
fam_ids = np.array(fam_ids_)[df_.subfamily.isin(subfam_sp_size_gteN)]
assert(n_cases==len(subfam_ids))
assert(n_cases==len(fam_ids))

# For text on visualization
font = ImageFont.truetype('arial.ttf', 20)

# Auto-Encoder
model = VSC(cdim=3, hdim=512, channels=str_to_list("32, 64, 128, 256, 512, 512"), image_size=256).cuda()
load_model(model, f'./pretrained/{model_id}.pth')
_ = model.eval()

# Family Predictor
class FamilyPrediction(nn.Module):

    def __init__(self, in_ch=1024, n_classes=97, n_neurons=256, n_layers=1, shrink_rate=1):
        super(FamilyPrediction, self).__init__()

        negative_slope = 0.2

        predict_family = [
            nn.Linear(in_ch, n_neurons),
            nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
        ]

        for _ in range(n_layers):
            predict_family.extend([
                nn.Dropout(0.5),
                nn.Linear(n_neurons, int(n_neurons//shrink_rate)),
                nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
            ])
            n_neurons //= shrink_rate
            n_neurons = int(n_neurons)

        predict_family.extend([
            nn.Linear(n_neurons, n_classes),
            nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
        ])

        self.predict_family =nn.Sequential(*predict_family)

    def forward(self, x):
        pred = self.predict_family(x)
        return pred

net = FamilyPrediction(in_ch=1024, n_neurons=256, n_layers=1, shrink_rate=1, n_classes=n_classes).cuda()
model_path = "./save/%s/subfam_pred_with_vsc_reparams_20220119.pkl" % profile_dir
state_dict = torch.load(model_path)
net.load_state_dict(state_dict)

# back propagation preparing
bp_app = BackPropApp(net)

# use back propagation to find out the effectiveness (gradients) of each dimension for subfamily predictions
# set to False if the gradients have been saved
rebuild_highlights = False
if rebuild_highlights:

    # init empty vectors
    highlights = np.empty([n_cases, code_dim])

    for selected_idx in range(n_cases):
        print(selected_idx, end='\r')

        # double the 512 dim into 1024 with no negative values
        selected = df.iloc[selected_idx:selected_idx+1,:].values
        selected_code = codes_to_no_neg(selected[:,:512].astype(float))

        # the inputs require gradients
        idv_code_var = Variable(torch.tensor(selected_code).float().cuda(), requires_grad=True)

        # get the gradients
        subfam_id = subfam_ids[selected_idx]
        cam = bp_app.get_cam(idv_code_var, subfam_id)
        cam_numpy = cam.data.cpu().numpy()
        highlights[selected_idx] = cam_numpy

    # aggregate on different levels and cached on the disk
    highlights_df = pd.DataFrame(highlights, index=None)
    highlights_df['subfamily'] = df.subfamily.values
    highlights_df['subfam_id'] = subfam_ids

    highlights_df['family'] = df.family.values
    highlights_df['fam_id'] = fam_ids

    highlights_df['sp'] = df.sp.values

    highlights_df.to_csv("./save/%s/highlights_no_neg.csv" % profile_dir, index=False)
    
    highlights_df_sp_mean = highlights_df.groupby(['subfamily', 'sp']).mean()
    highlights_df_sp_mean.to_csv("./save/%s/highlights_sp_mean_no_neg.csv" % profile_dir, index=False)

    highlights_df_subfam_mean = highlights_df_sp_mean.groupby('subfam_id').mean()
    highlights_df_subfam_mean = highlights_df_subfam_mean.reset_index('subfam_id')
    highlights_df_subfam_mean.to_csv("./save/%s/highlights_subfam_mean_no_neg.csv" % profile_dir, index=False)

    highlights_df_fam_mean = highlights_df_subfam_mean.groupby('fam_id').mean()
    highlights_df_fam_mean = highlights_df_fam_mean.reset_index('fam_id')
    highlights_df_fam_mean.drop(['subfam_id'], axis=1, inplace=True)
    highlights_df_fam_mean.to_csv("./save/%s/highlights_fam_mean_no_neg.csv" % profile_dir, index=False)
else:
    del net
    # load cached from the disk
    highlights_df = pd.read_csv("./save/%s/highlights_no_neg.csv" % profile_dir)
    highlights_df_sp_mean = pd.read_csv("./save/%s/highlights_sp_mean_no_neg.csv" % profile_dir)
    highlights_df_subfam_mean = pd.read_csv("./save/%s/highlights_subfam_mean_no_neg.csv" % profile_dir)
    highlights_df_fam_mean = pd.read_csv("./save/%s/highlights_fam_mean_no_neg.csv" % profile_dir)

# aggregate data on different levels
df_sp = df.copy()
df_sp_mean = df_sp.groupby(['subfamily', 'sp']).mean()
df_subfam_mean = df_sp_mean.groupby('subfam_id').mean()
df_subfam_mean = df_subfam_mean.reset_index('subfam_id')
df_fam_mean = df_subfam_mean.groupby('fam_id').mean()
df_fam_mean.drop(['subfam_id'], axis=1, inplace=True)
df_fam_mean = df_fam_mean.reset_index('fam_id')
# df_fam_mean.to_csv("./save/%s/df_fam_mean.csv" % profile_dir, index=False)



# prepare voting
# vote: select out the most effective N dimensions of each species in a given subfamily, 
# and then find out the N most shown-up dimensions as voted results
# here we try N from 1 to 20
minTopN = 1
maxTopN = 20 + minTopN
topN_features_voted_unions_length = []
topN_features_voted_unions = []
topN_features_voted = []
voted_to_unions_ratio = []

unique_subfam_ids = np.unique(highlights_df_sp_mean.subfam_id.values)

for topN_option in range(minTopN,maxTopN):
    
    subfam_feature_topN_by_voting = []
    for subfam_id in unique_subfam_ids:
        subfam_feature_topN_ = f(highlights_df_sp_mean[highlights_df_sp_mean.subfam_id == subfam_id].iloc[:,:code_dim].values, topN_option)
        subfam_feature_topN_by_voting.extend(f([np.histogram(subfam_feature_topN_, bins=code_dim, range=(0, code_dim))[0]], topN_option))

    features_voted = np.array(subfam_feature_topN_by_voting)

    topN_features_voted.append(features_voted)
    # the union (i.e. drop duplicated) of N voted features of subfamilies
    features_voted_unions = np.unique(np.concatenate(features_voted))
    topN_features_voted_unions_length.append(features_voted_unions.shape[0])
    topN_features_voted_unions.append(features_voted_unions)
    voted_to_unions_ratio.append(topN_option / features_voted_unions.shape[0])

###################################################### FIND THE BEST N (topN)
# use the unions of non-negative voted dimensions
# set the other dimensions to 0
# we evaluate the voted N with 3 grouping index with subfamiliy ids as labels

try:
    # if topN is not ready
    print(topN)
except:

    #minTopN = 1
    #maxTopN = 150 + 1
    #maxTopN = 50 + 1

    silhouettes = []
    calinski_harabaszs = []
    davies_bouldins = []

    df_remove_subfam_na = df[df.subfam_id.isin(np.unique(df_subfam_mean.subfam_id.values))]
    codes_remove_subfam_na_no_neg = codes_to_no_neg(df_remove_subfam_na.iloc[:,:512].values)
    #codes_remove_subfam_na_no_neg.shape

    for topN_option in range(minTopN, maxTopN):

        print(topN_option)

        shifted = topN_option - minTopN
        features_voted_unions = topN_features_voted_unions[shifted].copy()

        # df_picked = pd.DataFrame(codes_remove_subfam_na_no_neg).copy()
        # df_codes = df_picked.copy()

        df_codes = pd.DataFrame(codes_remove_subfam_na_no_neg)
        df_picked = pd.DataFrame(np.zeros_like(codes_remove_subfam_na_no_neg))

        voted = features_voted_unions
        print([topN_option, len(voted)])

        #df_picked.iloc[:,:] = 0
        df_picked.iloc[:, voted] = df_codes.iloc[:, voted]

        df_picked_sp = df_picked.copy()
        df_picked_sp['sp'] = df_remove_subfam_na.sp
        df_picked_sp['subfam_id'] = df.subfam_id
        df_picked_sp['fam_id'] = df.fam_id

        df_picked_sp_mean = df_picked_sp.groupby(['subfam_id', 'sp']).mean()
        df_picked_sp_mean = df_picked_sp_mean.reset_index('subfam_id')

        df_picked_sp_mean['subfam'] = subfams[df_picked_sp_mean.subfam_id]
        df_picked_sp_mean['family'] = fams_[df_picked_sp_mean.fam_id]

        color_map = (df_picked_sp_mean.subfam_id / np.max(df_picked_sp_mean.subfam_id + 1)).values
        df_picked_sp_mean_with_subfam_id = df_picked_sp_mean.copy()
        df_picked_sp_mean = df_picked_sp_mean.iloc[:,1:(1+code_dim)]

        # os.path.realpath(feature_abstract_path + '/sp_subfam_pc3.csv')

        silhouette = metrics.silhouette_score(df_picked_sp_mean, df_picked_sp_mean_with_subfam_id.subfam_id, metric='euclidean')
        calinski_harabasz = metrics.calinski_harabasz_score(df_picked_sp_mean, df_picked_sp_mean_with_subfam_id.subfam_id)
        davies_bouldin = metrics.davies_bouldin_score(df_picked_sp_mean, df_picked_sp_mean_with_subfam_id.subfam_id)

        silhouettes.append(silhouette)
        calinski_harabaszs.append(calinski_harabasz)
        davies_bouldins.append(davies_bouldin)

    # draw the grouping indices of evaluated N
    from matplotlib.ticker import MultipleLocator
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    minorLocatorX = MultipleLocator(1.0)
    minorLocatorY = MultipleLocator(0.05)

    silhouette_std = scaler.fit_transform(np.array(silhouettes).reshape(-1,1)).reshape(-1)
    calinski_harabasz_std = scaler.fit_transform(np.array(calinski_harabaszs).reshape(-1,1)).reshape(-1)
    davies_bouldin_std = 1 - scaler.fit_transform(np.array(davies_bouldins).reshape(-1,1)).reshape(-1)
    overall = (silhouette_std + calinski_harabasz_std + davies_bouldin_std) / 3

    plt.figure(figsize=(9,4))
    ax = plt.axes()

    ax.plot(np.arange(minTopN,maxTopN), silhouette_std, label='Silhouette')
    ax.plot(np.arange(minTopN,maxTopN), calinski_harabasz_std, label='Calinski Harabasz')
    ax.plot(np.arange(minTopN,maxTopN), davies_bouldin_std, label='1 - Davies Bouldin')
    ax.plot(np.arange(minTopN,maxTopN), overall, label='Overall')
    plt.xlabel("Number of features selected per subfamily")
    plt.ylabel("Scaled Index")
    ax.set_xticks(range(1, 21))
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    find_best_N_path = './save/%s' % profile_dir
    touch_dir(find_best_N_path)
    plt.savefig(find_best_N_path + f'/find_best_N_til_{maxTopN-1}.jpg')
    # plt.savefig(find_best_N_path + f'/find_best_N_til_{maxTopN-1}.pdf')

    topN = np.arange(minTopN,maxTopN)[np.argmax(overall)]
    # should be 2
    print(topN)

#######################################################################################################################
# preparing the data with topN=2 for the following analyses
topN = 2
shifted = topN - minTopN
features_voted = topN_features_voted[shifted].copy()
features_voted_unions = topN_features_voted_unions[shifted]
# should be 59
assert(len(features_voted_unions) == topN_features_voted_unions_length[shifted])
print(len(features_voted_unions))

################### no_neg to with_neg, throw errors on highlight conflicts
# The voted dimensions of the first 512 and the second 512 should be no overlap
features_voted_assertion = features_voted.copy()
features_voted_assertion[features_voted_assertion >= 512] = features_voted_assertion[features_voted_assertion >= 512] - 512
for i in range(len(features_voted_assertion)):
    try:
        assert(features_voted_assertion[i].shape[0] == np.unique(features_voted_assertion[i]).shape[0])
    except: 
        print(i)

############################################ subfamily-wised, data preparations
################### unions of picked features no neg
shifted = topN - minTopN
df_remove_subfam_na = df[df.subfam_id.isin(np.unique(df_subfam_mean.subfam_id.values))]
codes_remove_subfam_na_no_neg = codes_to_no_neg(df_remove_subfam_na.iloc[:,:512].values)

codes_remove_subfam_na_no_neg.shape
features_voted_unions = topN_features_voted_unions[shifted].copy()
len(features_voted_unions)

df_codes = pd.DataFrame(codes_remove_subfam_na_no_neg)
df_picked = pd.DataFrame(np.zeros_like(codes_remove_subfam_na_no_neg))

voted = features_voted_unions

voted_unique = voted.copy()
voted_unique[voted>=512] = voted_unique[voted>=512]-512
np.unique(voted_unique).shape
voted_unique

#df_picked.iloc[:,:] = 0
df_picked.iloc[:, voted] = df_codes.iloc[:, voted]

(df_picked<0).sum(axis=1).sum()

df_picked_sp = df_picked.copy()
df_picked_sp['sp'] = df_remove_subfam_na.sp
df_picked_sp['subfam_id'] = df.subfam_id
df_picked_sp['fam_id'] = df.fam_id

df_picked_sp_mean = df_picked_sp.groupby(['subfam_id', 'sp']).mean()
df_picked_sp_mean = df_picked_sp_mean.reset_index('subfam_id')

df_picked_sp_mean['subfam'] = subfams[df_picked_sp_mean.subfam_id]
df_picked_sp_mean['family'] = fams_[df_picked_sp_mean.fam_id]

df_picked_sp_mean_with_subfam_id = df_picked_sp_mean.copy()
color_map = (df_picked_sp_mean_with_subfam_id.subfam_id / np.max(df_picked_sp_mean_with_subfam_id.subfam_id + 1)).values
fam_color_map = (df_picked_sp_mean_with_subfam_id.fam_id / np.max(df_picked_sp_mean_with_subfam_id.fam_id + 1)).values

feature_abstract_path = './save/%s/top%d/' % (profile_dir, topN)
touch_dir(feature_abstract_path)

# cache the voted mean results
# df_picked_sp_mean_with_subfam_id.to_csv(feature_abstract_path + '/sp_no_neg_union_feat_mean.csv', index=True, sep='\t')
df_picked_sp_mean_with_subfam_id = pd.read_csv(feature_abstract_path + '/sp_no_neg_union_feat_mean.csv', sep='\t', index_col=0)

df_picked_sp_mean = df_picked_sp_mean_with_subfam_id.iloc[:,1:(1+code_dim)]

########### UMAP VISUALIZATION #############################################################

from umap import UMAP

# we color the result by family
# use the median number of species per family as n_neighbors

n_neighbors = int(df_picked_sp_mean_with_subfam_id.groupby('family').size().median())
umap = UMAP(n_neighbors=n_neighbors, n_components=2)
X_umap = umap.fit_transform(df_picked_sp_mean.loc[:, features_voted_unions.astype(str)])

# Run the lines only to get the top 20 families from the Goldstein2017 references and show label on the scatter plot
# Totally OK to ignore the labels
# fam_meta = pd.read_csv('save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/moth_geometry_with_meta.csv', sep='\t')
# topNfams = fam_meta.sort_values('Goldstein2017').iloc[-20:].family.values
fams_with_n_sp_top20 = pd.read_csv(f'fam_in_2019_gbif_WUver2_20220513.csv')[['family', 'Goldstein2017']].dropna().sort_values(by='Goldstein2017').values[-20:,0]

fig = plt.figure(figsize=(20, 20))
umap_texts = []
plt.scatter(X_umap[:,0], X_umap[:,1],
               color=plt.cm.jet(fam_color_map),
               s=20, edgecolor=None)
plt.xlabel('UMAP 1', {'fontsize':25})
plt.ylabel('UMAP 2', {'fontsize':25})
show_text = True
for fam_id in np.unique(df_picked_sp_mean_with_subfam_id.fam_id):               
    fam = fams_[fam_id]
    if fam not in fams_with_n_sp_top20:
        continue
    fam_matched_idx = np.where((df_picked_sp_mean_with_subfam_id.fam_id == fam_id))[0]
    fam_mean = X_umap[fam_matched_idx,:]
    if show_text:
        umap_texts.append(plt.text(x=fam_mean[:,0].mean(), y=fam_mean[:,1].mean(), s=fam, color='k', fontsize=12))

adjust_text(umap_texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='lightgray', lw=1))

# plt.show()

pca_path = './save/%s/top%d/pca_highlights' % (profile_dir, topN)
touch_dir(pca_path)

if show_text:
    plt.savefig(pca_path + f"/UMAP_nn{n_neighbors}_sp_subfam_picked_top{topN}_no_neg_union_colored_with_fam.pdf")
else:
    plt.savefig(pca_path + f"/UMAP_nn{n_neighbors}_sp_subfam_picked_top{topN}_no_neg_union_colored_with_fam_no_text.pdf")

plt.close()


# check for institutional sampling bias
# The Nymphalidae spcimen images from museum YPM have darker background
# Check the distribution of images from YPM and from other museums on a scatter plot to see if the backgroud effect the distribution
# They are basicly mixed together, meaning the background doesn't matter to the morphospace embeddings
df_inst = df.copy()
df_inst['is_YPM'] = 0
df_inst.loc[df.filename.str.contains('YPM'), 'is_YPM'] = 1
df_inst_sp_mean = df_inst.groupby('sp').mean()

# Global view
fig = plt.figure(figsize=(20, 15))
plt.scatter(X_umap[:,0], X_umap[:,1],
               color='lightgrey',
               s=20, edgecolor='k', alpha=.1)

fam_matched_idx = np.where((df_picked_sp_mean_with_subfam_id.family == 'Nymphalidae'))[0]
white_blue_colormap = (1-df_inst_sp_mean.is_YPM.values[fam_matched_idx])/2

plt.scatter(X_umap[fam_matched_idx][:,0], X_umap[fam_matched_idx][:,1],
               color=plt.cm.seismic(white_blue_colormap),
               s=20, lw=.2, edgecolor='k', alpha=.5)

plt.show()

# Zoom in
fig = plt.figure(figsize=(20, 15))
ax = p3.Axes3D(fig)
ax.view_init(54, -74)
ax.scatter(X_dr[fam_matched_idx][:,0], X_dr[fam_matched_idx][:,1], X_dr[fam_matched_idx][:,2],
               color=plt.cm.seismic(white_blue_colormap),
               s=20, edgecolor='k')
plt.show()

###################################################### Interpolation
# visualize the 12 interpolation forms of the most depart two species of each subfamily
def interpolate1d(codes2xD, N=5, D=512):
    codeDxN = np.empty([D, N])
    codesDx2 = codes2xD.transpose(1,0)
    for rid in range(codesDx2.shape[0]):
        code_t = codesDx2[rid]
        code_t_N = np.linspace(code_t[0], code_t[1], N)
        codeDxN[rid] = code_t_N
    codesNxD = codeDxN.transpose(1,0)
    return codesNxD

model = model.eval()
interpolated_path = "./save/%s/top%d/interpolated" % (profile_dir, topN)
touch_dir(interpolated_path)

#fam_jpg_fname = 'Erebidae.jpg'

with torch.no_grad():
    for subfam_id in df_subfam_mean.subfam_id.values:
        subfam_name = subfams[subfam_id]
        subfam_jpg_fname = subfam_name + '.jpg'
        codes = df_picked_sp_mean_with_subfam_id[df_picked_sp_mean_with_subfam_id.subfam_id==subfam_id].iloc[:,1:1025].values
        codes.shape
        # codes1 = np.max(codes, axis=0)
        # codes2 = np.min(codes, axis=0)
        pwdistances = pairwise_distances(codes)
        max_ids = np.argmax(pwdistances, axis=1)
        #max_ids.shape
        #codes.shape
        #rid = 2
        max_distance = -np.inf
        for rid, max_id in enumerate(max_ids):
            #max_pos = np.argmax(pwdistances[rid, np.argmax(pwdistances, axis=1)])
            #max_id = max_ids[max_pos]
            distance = pwdistances[rid, max_id]
            if distance > max_distance:
                max_distance = distance
                max_index0 = rid
                max_index1 = max_id

        # pwdistances[max_index0, max_index1]
        codesN = interpolate1d(codes_to_with_neg(np.array([codes[max_index0], codes[max_index1]])), N=12)
        codesN_cuda = torch.from_numpy(codesN).float().cuda()
        interpolated_imgs = model.decode(codesN_cuda)
        vutils.save_image(interpolated_imgs.data.cpu(), interpolated_path + '/' + subfam_jpg_fname, nrow=4)

######################### Calculate linkages of FAMILIES on 512D Original

from sklearn.metrics import pairwise_distances
import mpl_toolkits.mplot3d.axes3d as p3

topN_features_voted_unions[shifted]

#################################### 請超級注意 feature value 是從哪裡開始!!!!!!!!!!!!!!!! ################################################################
# dm_current_results = np.load("./save/%s/top%d/phylo_tree/hcluster_average/fam_distance_matrix.npy" % (profile_dir, topN))
# dm_current_results_minor_revised = np.load("./save/%s/top%d/phylo_tree/hcluster_average/fam_distance_matrix_minor_revised.npy" % (profile_dir, topN))

df_picked_sp_mean_dim_reduced = np.take_along_axis(df_picked_sp_mean.values, np.array([features_voted_unions]), axis=1)
df_picked_sp_mean_dim_reduced.shape

num_of_unique_group = len(df_fam_mean)

def build_phylo_tree(aff_linkage, linkage):

    print([aff_linkage, linkage])

    fam_aff = np.zeros([num_of_unique_group, num_of_unique_group])

    #i, j = 0, 0
    for i in range(num_of_unique_group):
        for j in range(i+1, num_of_unique_group):
            if i == j:
                # subfam_aff[i, j] = 0.
                pass
            else:
                cluster1 = df_picked_sp_mean_dim_reduced[df_sp_mean.fam_id==df_fam_mean.fam_id.iloc[i]]
                cluster2 = df_picked_sp_mean_dim_reduced[df_sp_mean.fam_id==df_fam_mean.fam_id.iloc[j]]

                if aff_linkage == 'ward':
                    # WARD
                    unions = np.concatenate([cluster1, cluster2])
                    centroid = unions.mean(axis=0)
                    aff = ((unions - centroid)**2).sum(axis=1).mean()
                elif aff_linkage == 'average':
                    # AVERAGE
                    aff = pairwise_distances(cluster1, cluster2).mean()
                elif aff_linkage == 'complete':
                    # COMPLETE
                    aff = pairwise_distances(cluster1, cluster2).max()
                elif aff_linkage == 'single':
                    # SINGLE
                    aff = pairwise_distances(cluster1, cluster2).min()

                fam_aff[i, j] = aff
                fam_aff[j, i] = aff

    affinity_matrix = fam_aff
    touch_dir("./save/%s/top%d/phylo_tree/hcluster_%s" % (profile_dir, topN, aff_linkage))
    np.save("./save/%s/top%d/phylo_tree/hcluster_%s/fam_distance_matrix_minor_revised.npy" % (profile_dir, topN, aff_linkage), affinity_matrix, allow_pickle=False)
    return affinity_matrix

#phylo_profile = 'weighted_no_neg_union'
#phylo_profile = 'original'
#linkage = 'average'
#aff_linkages = ['ward', 'average', 'complete', 'single']
#linkages = ['average', 'complete', 'single']

aff_linkages = ['average']
#aff_linkages = ['average']
linkages = ['none']

# aff_linkages = ['ward']
# linkages = ['average']

for aff_linkage in aff_linkages:
    for linkage in linkages:
        affmatrix = build_phylo_tree(aff_linkage, linkage)

affmatrix.shape

###################### Reduce mean fam feature to 1-D

df_picked_fam_mean_for_1d = df_picked_sp_mean_with_subfam_id.groupby(['subfam_id']).mean().groupby(['fam_id']).mean()
df_picked_fam_mean_for_1d.shape
df_picked_fam_mean_for_1d_voted = df_picked_fam_mean_for_1d.iloc[:,voted]
df_picked_fam_mean_for_1d_voted['family'] = fams_[df_picked_fam_mean_for_1d_voted.index]
df_picked_fam_mean_for_1d_voted

pca_for_1d = PCA()
pca_for_1d.fit(df_picked_fam_mean_for_1d)
X_dr_for_1d = pca_for_1d.transform(df_picked_fam_mean_for_1d)
pca_for_1d_explained_variance_ratio_cumsum = np.cumsum(pca_for_1d.explained_variance_ratio_)

pd.DataFrame(pca_for_1d.explained_variance_ratio_.reshape(-1, 4))
pca_for_1d_loadings = pca_for_1d.components_.T * np.sqrt(pca_for_1d.explained_variance_ratio_)
pd.DataFrame(np.round(pca_for_1d_loadings, 2)[features_voted_unions][:,:5]).to_csv('pca_for_1d_loadings.csv', sep='\t')
features_voted_unions[:20]
for f_ in list(features_voted_unions):
    print(f_)

X_dr_for_1d_df = pd.DataFrame(X_dr_for_1d)
X_dr_for_1d_df.shape
X_dr_for_1d_df['family'] = df_picked_fam_mean_for_1d_voted.family.values
X_dr_for_1d_df[['family', 0, 1, 2]].to_csv(pca_path + '/fam_mean/X_dr_for_1d.csv', index=False, sep='\t')

#fams_with_n_sp_top20 = df_picked_sp_mean_with_subfam_id.groupby('family').size().sort_values(ascending=False).iloc[:20].index.values
fams_with_n_sp_top20 = pd.read_csv(f'fam_in_2019_gbif_WUver2_20220513.csv')[['family', 'Goldstein2017']].dropna().sort_values(by='Goldstein2017').values[-20:,0]
# fam_meta

# project sp mean onto PCA-fam-mean space
sp_mean_on_fam_mean_pca = pca_for_1d.transform(df_picked_sp_mean)
sp_mean_on_fam_mean_pca_df = pd.DataFrame(sp_mean_on_fam_mean_pca)
sp_mean_on_fam_mean_pca_df['family'] = df_picked_sp_mean_with_subfam_id.family.values
sp_mean_on_fam_mean_pca_df[['family', 0, 1, 2]].to_csv(pca_path + '/fam_mean/sp_mean_on_fam_mean_pca.csv', index=False, sep='\t')


# plot fam pca
def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return int(width), int(height)

def hex_(n):
    if n < 16:
        return '0' + hex(n)[-1:]
    else:
        return hex(n)[-2:]

def get_fam_color (fam):
    if fam in ['Crambidae', 'Pyralidae', 'Alucitidae', 'Tortricidae', 'Sesiidae', 'Stathmopodidae', 'Ashinagidae', 'Lecithoceridae', 'Roeslerstammiidae', 'Adelidae', 'Carposinidae', 'Pterophoridae', 'Plutellidae', 'Ypsolophidae', 'Depressariidae', 'Ethmiidae', 'Oecophoridae', 'Gelechiidae', 'Tineidae', 'Glyphipterigidae']:
        # LightSteelBlue
        return '#' + hex_(122) + hex_(189) + hex_(199) #'tab:blue'
    elif fam in ['Erebidae', 'Cossidae', 'Metarbelidae', 'Sphingidae', 'Hepialidae', 'Notodontidae', 'Euteliidae', 'Noctuidae', 'Nolidae']:
        # DarkSeaGreen
        return '#' + hex_(152) + hex_(126) + hex_(167) #'tab:green'
    elif fam in ['Choreutidae', 'Hyblaeidae', 'Hesperiidae', 'Psychidae', 'Zygaenidae']:
        # LightSalmon
        return '#' + hex_(132) + hex_(158) + hex_(115) #'tab:orange'
    elif fam in ['Pieridae', 'Brahmaeidae', 'Eupterotidae', 'Drepanidae', 'Uraniidae', 'Geometridae', 'Endromidae', 'Sematuridae', 'Saturniidae', 'Mimallonidae', 'Bombycidae', 'Lasiocampidae', 'Limacodidae', 'Megalopygidae', 'Thyrididae']:
        # Moccasin
        return '#' + hex_(231) + hex_(199) + hex_(52) #return 'yellow'
    elif fam in ['Lycaenidae', 'Riodinidae', 'Papilionidae', 'Nymphalidae']:
        return '#' + hex_(253) + hex_(136) + hex_(137) #'Violet'
    else:
        print(fam)
        return '#' + hex_(225) + hex_(223) + hex_(205) #'White'

dpi = 300
ax_fontsize = 12

color_map_for_1d = [get_fam_color(f) for f in X_dr_for_1d_df.family]
fam_sp_num = df_picked_sp_mean_with_subfam_id.groupby('family').size().values
img_on_axis = False
#######################################
fam_sp_counts_df = pd.read_csv(f'fam_in_2019_gbif_WUver2_20220513.csv')[['family', 'Goldstein2017']]

fam_sp_counts = []
for fam_ in X_dr_for_1d_df.family.values:
    if fam_ not in fam_sp_counts_df.family.values:
        fam_sp_counts.append(fam_sp_counts_df.Goldstein2017.min())
    else:
        fam_sp_counts.append(fam_sp_counts_df[fam_sp_counts_df.family==fam_].Goldstein2017.values[0])
        
for x_axis in range(0,2):
    for y_axis in range(1,3):
        if y_axis > x_axis:
            texts = []
            print(x_axis, y_axis)
            plt.close(); fig_, ax_ = plt.subplots(figsize=(21, 21), dpi=dpi)
            #plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
            plt.xticks(fontsize=ax_fontsize)
            plt.yticks(fontsize=ax_fontsize)
            plt.xlabel(f'PC {x_axis+1}', fontsize=ax_fontsize+2)
            plt.ylabel(f'PC {y_axis+1}', fontsize=ax_fontsize+2)
            ax_width, ax_height = get_ax_size(fig_, ax_)
            # set figure span from -3 to 3
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.scatter(X_dr_for_1d[:,x_axis], X_dr_for_1d[:,y_axis], s=(np.sqrt(np.log(fam_sp_counts) * 1)).astype(int)*100, edgecolor='gray', linewidth=1, c=color_map_for_1d)
            # plt.scatter(X_dr_for_1d[:,x_axis], X_dr_for_1d[:,y_axis], s=(np.sqrt(np.log(fam_sp_meta.family_sp_count) * 1)).astype(int)*100, edgecolor='gray', linewidth=1, c=color_map_for_1d)
            # plt.scatter(X_dr_for_1d[:,x_axis], X_dr_for_1d[:,y_axis], edgecolor='gray', linewidth=1, c=color_map_for_1d)
            for _, row in X_dr_for_1d_df.iterrows():
                if row.family in fams_with_n_sp_top20:
                    texts.append(plt.text(row.iloc[x_axis], row.iloc[y_axis], row.family, fontsize=20))
            adjust_text(texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='lightgray', lw=1))
            plt.savefig(pca_path + f'/fam_mean/fam_mean_pca_h7_pc{x_axis+1}{y_axis+1}_top20fam.pdf')
            plt.savefig(pca_path + f'/fam_mean/fam_mean_pca_h7_pc{x_axis+1}{y_axis+1}_top20fam.png')
            plt.close()

#######################################
##### plotly 3d plot

import plotly.graph_objects as go

is_in_top20 = X_dr_for_1d_df.family.isin(fams_with_n_sp_top20).values

fig = go.Figure(data=[go.Scatter3d(
    x=X_dr_for_1d[:,0],
    y=X_dr_for_1d[:,1],
    z=X_dr_for_1d[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=color_map_for_1d,
        opacity=0.8,
        line=dict(
            color='black',
            width=1
        )
    ),
    text=X_dr_for_1d_df.family.values
), go.Scatter3d(
    x=X_dr_for_1d[is_in_top20,0],
    y=X_dr_for_1d[is_in_top20,1],
    z=X_dr_for_1d[is_in_top20,2],
    mode='text',
    text=X_dr_for_1d_df.family.values[is_in_top20],
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig.write_html(f'./save/vsc/epoch_{epoch_num}/20211230_g{num_cls}_{epoch_num}_{cls_method}3d.html')
#fig.show(renderer="iframe")
fig.show()

######################################################################## 4-corner Interpolation on fam-mean pca

val_range = 3
eps = 1e-10
steps = 3
pca_axis_step = val_range / steps
num_of_steps = int(2 * val_range / pca_axis_step) + 1
nrow = num_of_steps

topleft = np.array([-3, 3])
topright = np.array([3, 3])
bottomleft = np.array([-3, -3])
bottomright = np.array([3, -3])

override = True
if override:
    num_of_steps = 4
    nrow = num_of_steps

toprowpoints = np.array([topleft + i * (topright - topleft) / (num_of_steps - 1) for i in range(num_of_steps)])
bottomrowpoints = np.array([bottomleft + i * (bottomright - bottomleft) / (num_of_steps - 1) for i in range(num_of_steps)])

fourcorner_interpols = np.array([toprowpoints + i * (bottomrowpoints - toprowpoints) / (num_of_steps - 1) for i in range(num_of_steps)])
fourcorner_interpols.shape

# change the values to show different pc axis
# [0, 1] pc1, pc2
# [0, 2] pc1, pc3
# [1, 2] pc2, pc3
fill_in_idx = [1, 2]
no_neg_codes_ = []
for interpols_ in fourcorner_interpols.reshape(-1, 2):
    pca_dr_code_ = np.zeros(pca_for_1d.components_.shape[0])
    #print(interpols_)
    pca_dr_code_[fill_in_idx[0]] = interpols_[0]
    pca_dr_code_[fill_in_idx[1]] = interpols_[1]
    no_neg_codes_.append(pca_for_1d.inverse_transform(pca_dr_code_.reshape(1,-1)).reshape(-1))

with torch.no_grad():
    fourcorner_interpol_imgs_cuda = model.decode(torch.from_numpy(codes_to_with_neg(np.array(no_neg_codes_))).float().cuda())

touch_dir(pca_path + f'/fam_mean/PC_axis{fill_in_idx[0]+1}{fill_in_idx[1]+1}_{num_of_steps}x{num_of_steps}')
fourcorner_interpol_imgs = save_image(fourcorner_interpol_imgs_cuda, pca_path + f'/fam_mean/PC_axis{fill_in_idx[0]+1}{fill_in_idx[1]+1}_{num_of_steps}x{num_of_steps}/pca_axis{fill_in_idx[0]+1}{fill_in_idx[1]+1}_four_corner_{num_of_steps}x{num_of_steps}_interpols.png', nrow=nrow, return_image=True)

fourcorner_interpol_imgs_npy = fourcorner_interpol_imgs_cuda.permute(0,2,3,1).data.cpu().numpy()

test_parameters = [-3, -1, 1, 3]
for i_ in range(16):
    x_idx_ = test_parameters[i_ % 4 ]
    y_idx_ = -test_parameters[i_ // 4]
    if x_idx_ in test_parameters and y_idx_ in test_parameters:
        print(x_idx_, y_idx_)
        img_xy = fourcorner_interpol_imgs_npy[i_]
        touch_dir(pca_path + f'/fam_mean/PC_axis{fill_in_idx[0]+1}{fill_in_idx[1]+1}_{num_of_steps}x{num_of_steps}')
        _ = io.imsave(pca_path + f'/fam_mean/PC_axis{fill_in_idx[0]+1}{fill_in_idx[1]+1}_{num_of_steps}x{num_of_steps}/xy_{x_idx_}_{y_idx_}.png', img_xy)


################################# Feat Family PhyloTree
fams = np.unique(df.family)
#fam_to_tree_dm = DistanceMatrix(np.load("./save/%s/top%d/phylo_tree/hcluster_average/fam_distance_matrix.npy" % (profile_dir, topN)), fams)
fam_to_tree_dm = DistanceMatrix(np.load("./save/%s/top%d/phylo_tree/hcluster_average/fam_distance_matrix_minor_revised.npy" % (profile_dir, topN)), fams)

fam_nj_tree = nj(fam_to_tree_dm)
fam_root_at = "Eriocraniidae"

fam_nj_tree_rooted = fam_nj_tree.root_at(fam_nj_tree.find(fam_root_at).parent)
# nj_tree_rooted = nj_tree.root_at(nj_tree.find('Zygaenidae').parent)

fam_phylo_nj_tree_rooted = Phylo.read(StringIO(str(fam_nj_tree_rooted)), "newick")
fam_phylo_nj_tree_rooted.ladderize()

colors = ["#808080", "#556b2f", "#7f0000", "#483d8b", "#008000", "#3cb371", "#008080", "#cd853f", "#000080", "#32cd32", "#8b008b", "#b03060", "#ff0000", "#ff8c00", "#00ff00", "#00fa9a", "#8a2be2", "#dc143c", "#00ffff", "#00bfff", "#0000ff", "#adff2f", "#d8bfd8", "#ff00ff", "#1e90ff", "#f0e68c", "#fa8072", "#ffff54", "#ff1493", "#7b68ee", "#afeeee", "#ee82ee", "#ffdab9"]

# we only need the fam subfam info for coloring
supfam_fam_subfam = pd.read_csv('supfam_fam_subfam_fill_average.txt', sep='\t')
supfam_fam_subfam['supfam_id'] = np.unique(supfam_fam_subfam.supfam_ours, return_inverse=True)[1]

def fam_get_tip_label2(clade):
    if clade.name is not None:
        supfam = None
        try:
            supfam = supfam_fam_subfam[supfam_fam_subfam.fam_ours==clade.name].supfam_ours.values[0]
        except:
            pass
        if supfam is not None:
            return supfam + '_' + clade.name
    return clade.name

def fam_get_tip_color2(label):
    try:
        return colors[supfam_fam_subfam[supfam_fam_subfam.fam_ours==label.split('_')[1]].supfam_id.values[0]]
    except:
        print('Oops', label)
        return 'black'

fig, axes = plt.subplots(1, 1, figsize=(40, 40), dpi=600)
Phylo.draw(fam_phylo_nj_tree_rooted, axes=axes, do_show=False, label_func=fam_get_tip_label2, label_colors=fam_get_tip_color2)
# Phylo.draw(fam_phylo_nj_tree_rooted, do_show=True, label_func=fam_get_tip_label2, label_colors=fam_get_tip_color2)
plt.savefig('./save/%s/top%d/phylo_tree/hcluster_average/phylo_nj_tree_rooted_Eriocraniidae_20220512.pdf' % (profile_dir, topN))
plt.clf()
plt.close()

for idx, c in enumerate(fam_phylo_nj_tree_rooted.find_clades()):
    if c.name:
        pass
    else:
        c.name = 'clade_%d' % idx

print(fam_phylo_nj_tree_rooted)
Phylo.write(fam_phylo_nj_tree_rooted, './save/%s/top%d/phylo_tree/hcluster_average/phylo_nj_tree_rooted_Eriocraniidae.nwk' % (profile_dir, topN), 'newick')
#Phylo.write(phylo_nj_tree_rooted, './save/%s/../../phylo_tree/hcluster_average/phylo_nj_tree_rooted_%s.nwk' % (profile_dir, root_at), 'newick')

clades = [clade for clade in fam_phylo_nj_tree_rooted.find_clades()]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0.5, 1.5))

################################### Generate typical forms for each clades of the morpho tree
#phylo_profile = 'no_neg_union'
tree_name = 'nj_fam_feat_tree'
sp_nearest = False
if sp_nearest:
    nj_clade_viz_dir = './save/%s/top%d/phylo_tree/hcluster_average/nj/clade_viz/%s_nearest_sp/' % (profile_dir, topN, tree_name)
    #nj_clade_viz_dir = './save/%s/../../phylo_tree/hcluster_average/nj/clade_viz/%s_nearest_sp/' % (profile_dir, tree_name)
else:
    #nj_clade_viz_dir = './save/%s/top%d/phylo_tree/hcluster_average/nj/clade_viz/%s/' % (profile_dir, topN, tree_name)
    nj_clade_viz_dir = './save/%s/top%d/phylo_tree/hcluster_average/nj/clade_viz/%s/' % (profile_dir, topN, tree_name)

touch_dir(nj_clade_viz_dir)

df_picked_fam_mean_no_neg_ = codes_to_no_neg(df_fam_mean.iloc[:,1:].values.copy())
df_picked_fam_mean_no_neg = np.zeros_like(df_picked_fam_mean_no_neg_)
for row_id in range(df_fam_mean.shape[0]):
    df_picked_fam_mean_no_neg[row_id][features_voted_unions] = df_picked_fam_mean_no_neg_[row_id][features_voted_unions]
df_picked_fam_mean = codes_to_with_neg(df_picked_fam_mean_no_neg)

for clade in clades:

    leaves_of_clade = clade.get_terminals()
    leaf_ids = [np.where(fams_ == leaf.name)[0][0] for leaf in leaves_of_clade]

    lead_id_iloc = [np.where(df_fam_mean.fam_id == lid)[0][0] for lid in leaf_ids]

    ### clade subfam mean
    #subfam_of_clade = df_subfam_mean.iloc[lead_id_iloc,1:].values
    fam_of_clade = df_picked_fam_mean[lead_id_iloc]

    if fam_of_clade.shape[0] == 1:
        g_code_ = fam_of_clade[0]
    else:
        dist_of_clade_to_each_fam = np.array([clade.distance(leaf.name) for leaf in leaves_of_clade])
        if dist_of_clade_to_each_fam.sum() == 0:
            dist_of_clade_to_each_fam = np.array([len(clade.get_path(leaf.name)) for leaf in leaves_of_clade])
        dist_of_clade_to_each_fam_scaled = scaler.fit_transform(dist_of_clade_to_each_fam.reshape(-1,1)).reshape(-1)
        m_top = dist_of_clade_to_each_fam_scaled.prod()
        m_bottom_ = m_top / dist_of_clade_to_each_fam_scaled
        m_bottom = m_bottom_.sum()
        m = m_top / m_bottom

        g_code_ = ((m / dist_of_clade_to_each_fam_scaled).reshape(-1,1) * fam_of_clade).sum(axis=0)

    ### nearest sp point to the average
    if sp_nearest:
        sp_of_clade = df_sp_mean[[fid in leaf_ids for fid in df_sp_mean.fam_id]].iloc[:,:512]
        g_code = sp_of_clade.values[np.argmin(pairwise_distances(np.array([g_code_]), sp_of_clade.values))]
    ### or just use the mean value
    else:
        g_code = g_code_
    
    g_codes_npy = np.array([g_code])
    g_codes_cuda = torch.from_numpy(g_codes_npy).float().cuda()
    with torch.no_grad():
        imgs = model.decode(g_codes_cuda)
    imgs_numpy = imgs.data.cpu().numpy().transpose(0,2,3,1)

    #resized_img = cv2.resize(imgs_numpy[0], (256, 171), interpolation=cv2.INTER_AREA)
    resized_img = imgs_numpy[0]

    io.imsave(nj_clade_viz_dir + '/%s.jpg' % clade.name, resized_img)


################################################################################
########################################## Grouping clades #####################

def get_node_list(clade):
    node_list = []
    for node in clade.find_clades():
        node_list.append(node.name)
    return node_list

# find all permutations of K non-zero integers that sum to S recursively
def find_all_permutations(K, S):
    if K == 1:
        return [[S]]
    elif K >= 2:
        return [[i] + p for i in range(1, S) for p in find_all_permutations(K-1, S-i)]

# remove intersection of 2 lists
def list_difference(list1, list2):
    return [x for x in list1 if x not in list2]

def whiskers_min_and_max(x_):
    x = x_.reshape(-1)
    if x.shape[0] < 1:
        return np.nan, np.nan, np.nan
    elif x.shape[0] == 1:
        return x[0], x[0], np.nan
    else:
        q1 = np.quantile(x, .25)
        q3 = np.quantile(x, .75)
        iqr = q3 - q1
        whiskers_min = max(q1 - 1.5 * iqr, np.min(x))
        whiskers_max = min(q3 + 1.5 * iqr, np.max(x))
        
        return whiskers_min, whiskers_max, whiskers_max - whiskers_min

def nan_normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


# get all clades on the right side of the tree
is_right = True
right_clades = []
ignore_list = []
for idx, c in enumerate(fam_phylo_nj_tree_rooted.find_clades()):
    print(c.name)
    if len(ignore_list) > 0:
        if c.name in ignore_list:
            continue
    if c.name.startswith('clade_'):
        if not is_right:
            ignore_list = get_node_list(c)
            is_right = True
        else:
            right_clades.append(c)
            is_right = False
    else:
        if not is_right:
            is_right = True
        else:
            pass

# get all permutations
K = 6
permutation_clades = []
for permutations in find_all_permutations(K, len(right_clades)):
    permutation_clades_ = []
    clade_counter = 0
    for i in permutations:
        permutation_clades_.append(right_clades[clade_counter])
        clade_counter += i
    permutation_clades.append(permutation_clades_)

k_counts_stds = []
k_sims_means = []
k_sims_stds = []
k_io_ratios_means = []
k_io_ratios_stds = []
for permutation_clades_ in permutation_clades:
    k_names = np.empty(K, dtype=object)
    for k_ in range(1, K):
        k_names[k_ - 1] = list_difference([c.name for c in permutation_clades_[k_ - 1].get_terminals()], [c.name for c in permutation_clades_[k_].get_terminals()])
    k_names[K - 1] = [c.name for c in permutation_clades_[K - 1].get_terminals()]

    k_counts = []
    k_sims = []
    k_io_ratios = []
    for k_name in k_names:
        k_ids = [np.where(fams == k_name_)[0][0] for k_name_ in k_name]
        not_k_ids = [np.where(fams == k_name_)[0][0] for k_name_ in fams if k_name_ not in k_name]
        k_ids_len = len(k_ids)
        k_counts.append(len(k_ids))
        i_dm = fam_to_tree_dm.data[k_ids][:, k_ids]
        i_dm_ut = i_dm[np.triu_indices(k_ids_len, 1)]
        k_sim = np.mean(i_dm_ut)
        #k_sim = whiskers_min_and_max(i_dm_ut)[2]
        # k_sim = i_dm.sum() / (k_ids_len * (k_ids_len - 1)) / 2
        k_sims.append(k_sim)

        o_dm = fam_to_tree_dm.data[k_ids][:, not_k_ids]
        o_sim = np.mean(o_dm)
        k_io_ratios.append(k_sim / o_sim)
        #k_io_ratios.append(k_sim / whiskers_min_and_max(o_dm)[2])

    k_counts_stds.append(np.std(k_counts))
    k_sims_means.append(np.mean(k_sims))
    k_sims_stds.append(np.std(k_sims))
    k_io_ratios_means.append(np.mean(k_io_ratios))
    k_io_ratios_stds.append(np.std(k_io_ratios))

# 各組差異最小化、總組內變異最小化、總[組內/組外]變異最小化
# minimize the total variances of all groups, minimize the total intra/inter variances of groups, minimize the inter-group characteristic difference
overall_index = \
    nan_normalize(np.array(k_sims_stds)) + \
    nan_normalize(np.array(k_io_ratios_stds)) + \
    nan_normalize(np.array(k_sims_means) + 2 * np.array(k_sims_stds)) + \
    nan_normalize(np.array(k_io_ratios_means) + 2 * np.array(k_io_ratios_stds))

##### Rerun the upper codes for K from 3 to 10 to find the minimum (optimum) result, which is when K=6
pid_optim = np.nanargmin(overall_index)

plt.figure(figsize=(15, 10))
plt.ylim(0, 4)
plt.plot(overall_index)
plt.axvline(pid_optim, color='r')
plt.text(pid_optim, np.nanmin(overall_index), np.round(np.nanmin(overall_index), 4))
plt.savefig(f'./save/vsc_epoch_40000/pred_fam/no_neg/top2/phylo_tree/hcluster_average/overall_index_{K}groups_min.png')
#plt.show()
plt.close()


########################################## Visualizing the grouping results
# get the fam names in each group
permutation_clades_optim = permutation_clades[pid_optim]
k_names_optim = np.empty(K, dtype=object)
for k_ in range(1, K):
    k_names_optim[k_ - 1] = list_difference([c.name for c in permutation_clades_optim[k_ - 1].get_terminals()], [c.name for c in permutation_clades_optim[k_].get_terminals()])

k_names_optim[K - 1] = [c.name for c in permutation_clades_optim[K - 1].get_terminals()]

df_picked_sp_mean_dim_reduced_with_group_id = pd.DataFrame(np.take_along_axis(df_picked_sp_mean.values, np.array([features_voted_unions]), axis=1))
df_picked_sp_mean_dim_reduced_with_group_id['group_id'] = 0
#group_i = 0
for group_i in range(K):
    for f_ in k_names_optim[group_i]:
        df_picked_sp_mean_dim_reduced_with_group_id.loc[df_picked_sp_mean_with_subfam_id.family.values == f_, 'group_id'] = group_i

df_picked_sp_mean_dim_reduced_with_group_id

n_neighbors_fam = int(df_picked_sp_mean_with_subfam_id.groupby('family').size().median())
umap_kgroup = UMAP(n_neighbors=n_neighbors_fam, n_components=2)

Kgroup_umap = umap_kgroup.fit_transform(df_picked_sp_mean_dim_reduced_with_group_id.values[:,:len(features_voted_unions)])

# df_picked_sp_mean.loc[:,features_voted_unions]

def hex_(n):
    if n < 16:
        return '0' + hex(n)[-1:]
    else:
        return hex(n)[-2:]

def get_group_color (gid):
    if gid == 1:
        # LightSteelBlue
        return '#' + hex_(122) + hex_(189) + hex_(199) #'tab:blue'
    elif gid == 2:
        # DarkSeaGreen
        return '#' + hex_(152) + hex_(126) + hex_(167) #'tab:green'
    elif gid == 3:
        # LightSalmon
        return '#' + hex_(132) + hex_(158) + hex_(115) #'tab:orange'
    elif gid == 4:
        # Moccasin
        return '#' + hex_(231) + hex_(199) + hex_(52) #return 'yellow'
    elif gid == 5:
        return '#' + hex_(253) + hex_(136) + hex_(137) #'Violet'
    else:
        return '#' + hex_(225) + hex_(223) + hex_(205) #'White'

fam_group_colors = np.array([get_group_color(group_id) for group_id in df_picked_sp_mean_dim_reduced_with_group_id.group_id.values])

fig = plt.figure(figsize=(10, 10), dpi=300)
plt.scatter(Kgroup_umap[:,0], Kgroup_umap[:,1],
               color=fam_group_colors,
               s=20, edgecolor=None)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('UMAP 1', {'fontsize':8})
plt.ylabel('UMAP 2', {'fontsize':8})
plt.savefig(f'./save/vsc_epoch_40000/pred_fam/no_neg/top2/phylo_tree/hcluster_average/coloring_{K}groups.pdf')
plt.show()

# get family names of each group
diff_groups = []
for k_ in range(1, len(k_names_optim)):
    diff_groups.append(list_difference(k_names_optim[k_-1], k_names_optim[k_]))
diff_groups.append(list_difference(k_names_optim[k_], []))

# should display the unprocessed root 'Eriocraniidae' and the family 'Cosmopterigidae' on the left
for group_id_ in range(6):
    print(set(df_picked_sp_mean_with_subfam_id.reset_index()[df_picked_sp_mean_dim_reduced_with_group_id.group_id==group_id_].family.unique()) - set(diff_groups[group_id_]))

fam_sp_meta = df.groupby(['family', 'sp']).head(1).groupby(['family']).size().reset_index().rename(columns={0: 'family_sp_count'})
fam_sp_meta['k_names_optim'] = 0

##############
# the column name `k_names_optim` is changed to `6groups` in the following moth geometry analysis
for k_ in range(0, len(k_names_optim)):
    for name in k_names_optim[k_]:
        fam_sp_meta.loc[fam_sp_meta.family == name, 'k_names_optim'] = k_

##################################### calculate the whisker Max dist of each subfamily
fam_sp_meta.to_csv('./save/%s/top%d/phylo_tree/hcluster_average/fam_sp_meta_20230825.csv' % (profile_dir, topN), sep='\t', index=False)
