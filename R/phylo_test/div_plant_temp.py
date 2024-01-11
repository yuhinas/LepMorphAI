import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def interpolate_value(x, Xarr, Yarr):
    """
    Interpolates a value y for a given scalar x based on arrays Xarr and Yarr.
    
    Parameters:
    x (scalar): The point at which to interpolate.
    Xarr (numpy array): Array of x-values.
    Yarr (numpy array): Array of y-values corresponding to Xarr.

    Returns:
    float: Interpolated value at x.
    """
    # Check if x is within the range of Xarr
    if x < np.min(Xarr) or x > np.max(Xarr):
        return None

    # Create the interpolation function
    interpolation_function = interp1d(Xarr, Yarr, kind='linear')

    # Interpolate the value
    y = interpolation_function(x)
    return y

svd_df = pd.read_csv('./phylo_test/stomata_vein_density/pbio.2003706.s003.csv')
ma_dtt_df = pd.read_csv('./phylo_test/ma_dtt_pcs.csv')
netdiv_df = pd.read_csv('./phylo_test/netdiv_time_xy.csv')
paleotmp_df = pd.read_csv('./phylo_test/paleotemperature/paleotemperature_1ma_unit.csv', sep='\t')

#svd_df['Dv_mm-1'].dropna()

netdiv_df_a1ma = netdiv_df[netdiv_df.times >= 50]
netdiv_align = netdiv_df.copy()
netdiv_align = netdiv_align.rename({'times': 'age', 'avg_NetDiv':'net_div_rate'}, axis=1)
# netdiv_align.to_csv('../LepMorphAI/phylo_test/netdivrate_dropna.csv', index=False)
# netdiv_df_a1ma.avg_NetDiv = netdiv_df_a1ma.avg_NetDiv / netdiv_df_a1ma.avg_NetDiv.max()

tmp_from = np.ceil(max(svd_df.age.max(), ma_dtt_df.Ma.max() * 100, netdiv_df.times.max()))
paleotmp_df_a1ma = paleotmp_df[(paleotmp_df.Age >= 50) & (paleotmp_df.Age <= tmp_from)]
# paleotmp_df_a1ma.GAT = paleotmp_df_a1ma.GAT / paleotmp_df_a1ma.GAT.max()
paleotmp_df_align = paleotmp_df.copy()
paleotmp_df_align = paleotmp_df_align.rename({'Age':'age', 'GAT':'temp'}, axis=1)
#paleotmp_df_align.to_csv('../LepMorphAI/phylo_test/paleotemp_dropna.csv', index=False)

ma_dtt_df_a1ma = ma_dtt_df.copy()
ma_dtt_df_a1ma.Ma = ma_dtt_df_a1ma.Ma * 100
ma_dtt_df_a1ma = ma_dtt_df_a1ma.rename({'Ma':'age', 'DTT_PC1':'dtt_pc1', 'DTT_PC2':'dtt_pc2', 'DTT_PC3':'dtt_pc3'}, axis=1)
ma_dtt_df_align = ma_dtt_df_a1ma.iloc[:-1, 1:]
#ma_dtt_df_align.to_csv('../LepMorphAI/phylo_test/dtt_dropna.csv', index=False)

# svd_df_a1ma = svd_df[(svd_df.age<=100) & (svd_df.age>=50)]
# clade_filter = svd_df.clade.unique()[0]
# clade_filter
svd_df_clade_median = svd_df.groupby(['age', 'clade']).median().reset_index()
svd_df_a1ma = svd_df_clade_median[(svd_df_clade_median.age>=50)]# & (svd_df_grouped.clade == clade_filter)]
svd_df_a1ma = svd_df_a1ma.sort_values('age')

sd_df_nona = svd_df_a1ma[['age', 'clade', 'Ds_mm-2']].dropna()
sd_df_nona_median = sd_df_nona.groupby('age').median().reset_index()
sd_df_nona_align = sd_df_nona_median.rename({'Ds_mm-2':'stomata_density'}, axis=1)

# sd_plot_mean_df['Ds_mm-2'] = sd_plot_mean_df['Ds_mm-2'] / sd_plot_mean_df['Ds_mm-2'].max()
# sd_df_dropna = svd_df[['age', 'clade', 'Ds_mm-2']].dropna()
# sd_df_dropna.rename({'Ds_mm-2':'stomata_density'}, axis=1).to_csv('../LepMorphAI/phylo_test/sd_dropna.csv', index=False)

svd_df

vd_df_nona = svd_df_a1ma[['age', 'clade', 'Dv_mm-1']].dropna()
vd_df_nona_median = vd_df_nona.groupby('age').median().reset_index()
vd_df_nona_align = vd_df_nona_median.rename({'Dv_mm-1':'vein_density'}, axis=1)

# vd_plot_df = svd_df_a1ma[['age', 'Dv_mm-1']].dropna()
# vd_plot_mean_df = vd_plot_df.groupby('age').mean().reset_index()
# # vd_plot_mean_df['Dv_mm-1'] = vd_plot_mean_df['Dv_mm-1'] / vd_plot_mean_df['Dv_mm-1'].max()
# vd_df_dropna = svd_df[['age', 'clade', 'Dv_mm-1']].dropna()
# vd_df_dropna.rename({'Dv_mm-1':'vein_density'}, axis=1).to_csv('../LepMorphAI/phylo_test/vd_dropna.csv', index=False)


# Intersection of ages
age_intersectoin = set.intersection(
    set(np.arange(np.ceil(sd_df_nona_align.age.min()), np.floor(sd_df_nona_align.age.max()) + 1, dtype=int)),
    set(np.arange(np.ceil(vd_df_nona_align.age.min()), np.floor(vd_df_nona_align.age.max()) + 1, dtype=int)),
    set(np.arange(np.ceil(netdiv_align.age.min()), np.floor(netdiv_align.age.max()) + 1, dtype=int)),
    set(np.arange(np.ceil(paleotmp_df_align.age.min()), np.floor(paleotmp_df_align.age.max()) + 1, dtype=int)),
    set(np.arange(np.ceil(ma_dtt_df_align.age.min()), np.floor(ma_dtt_df_align.age.max()) + 1, dtype=int)))
age_intersectoin = list(age_intersectoin)

# Union of ages
age_union = set.union(
    set(np.round(sd_plot_mean_df.age.values).astype(int)),
    set(np.round(vd_plot_mean_df.age.values).astype(int)),
    set(np.round(netdiv_df_a1ma.times.values).astype(int)),
    set(np.round(ma_dtt_df_a1ma.Ma.values).astype(int)),
)
age_union = list(age_union)

# Standardizing
# Stomata density
# ages = np.arange(np.ceil(sd_plot_mean_df.age.min()), np.floor(sd_plot_mean_df.age.max()) + 1, dtype=int)
ages = age_intersectoin
ages = age_union
sd_interpolateds = []
for t in ages:
    sd_interpolateds.append(interpolate_value(t, sd_df_nona_align.age.values, sd_df_nona_align['stomata_density'].values))
sd_interpolated_df = pd.DataFrame({'age':ages, 'val':sd_interpolateds})

# Vein density
# ages = np.arange(np.ceil(vd_plot_mean_df.age.min()), np.floor(vd_plot_mean_df.age.max()) + 1, dtype=int)
# ages = age_intersectoin
vd_interpolateds = []
for t in ages:
    vd_interpolateds.append(interpolate_value(t, vd_df_nona_align.age.values, vd_df_nona_align['vein_density'].values))
vd_interpolated_df = pd.DataFrame({'age':ages, 'val':vd_interpolateds})

# DTT PC1
# ages = np.arange(np.ceil(ma_dtt_df_a1ma.Ma.min()), np.floor(ma_dtt_df_a1ma.Ma.max()) + 1, dtype=int)
# ages = age_intersectoin
dtt_pc1_interpolateds = []
for t in ages:
    dtt_pc1_interpolateds.append(interpolate_value(t, ma_dtt_df_align.age.values, ma_dtt_df_align['dtt_pc1'].values))
dtt_pc1_interpolated_df = pd.DataFrame({'age':ages, 'val':dtt_pc1_interpolateds})

# DTT PC2
# ages = np.arange(np.ceil(ma_dtt_df_a1ma.Ma.min()), np.floor(ma_dtt_df_a1ma.Ma.max()) + 1, dtype=int)
# ages = age_intersectoin
dtt_pc2_interpolateds = []
for t in ages:
    dtt_pc2_interpolateds.append(interpolate_value(t, ma_dtt_df_align.age.values, ma_dtt_df_align['dtt_pc2'].values))
dtt_pc2_interpolated_df = pd.DataFrame({'age':ages, 'val':dtt_pc2_interpolateds})

# DTT PC3
# ages = np.arange(np.ceil(ma_dtt_df_a1ma.Ma.min()), np.floor(ma_dtt_df_a1ma.Ma.max()) + 1, dtype=int)
# ages = age_intersectoin
dtt_pc3_interpolateds = []
for t in ages:
    dtt_pc3_interpolateds.append(interpolate_value(t, ma_dtt_df_align.age.values, ma_dtt_df_align['dtt_pc3'].values))
dtt_pc3_interpolated_df = pd.DataFrame({'age':ages, 'val':dtt_pc3_interpolateds})

def global_min_max_scaling (X):
    x_max = np.max(X)
    x_min = np.min(X)
    return (X - x_min) / (x_max - x_min)

dtt_pcs_scaled = global_min_max_scaling(np.hstack([dtt_pc1_interpolated_df.val.values[:,None], dtt_pc2_interpolated_df.val.values[:,None], dtt_pc3_interpolated_df.val.values[:,None]]))

dtt_pc1_scaled = dtt_pcs_scaled[:,0]
dtt_pc2_scaled = dtt_pcs_scaled[:,1]
dtt_pc3_scaled = dtt_pcs_scaled[:,2]

# NET DIV RATE
# ages = np.arange(np.ceil(netdiv_df_a1ma.times.min()), np.floor(netdiv_df_a1ma.times.max()) + 1, dtype=int)
# ages = age_intersectoin
netdiv_interpolateds = []
for t in ages:
    netdiv_interpolateds.append(interpolate_value(t, netdiv_align.age.values, netdiv_align['net_div_rate'].values))
netdiv_interpolated_df = pd.DataFrame({'age':ages, 'val':netdiv_interpolateds})

# PALEOTMP
# ages = np.arange(np.ceil(paleotmp_df_a1ma.Age.min()), np.floor(paleotmp_df_a1ma.Age.max()) + 1, dtype=int)
# ages = age_intersectoin
paleotmp_interpolateds = []
for t in ages:
    paleotmp_interpolateds.append(interpolate_value(t, paleotmp_df_align.age.values, paleotmp_df_align['temp'].values))
paleotmp_interpolated_df = pd.DataFrame({'age':ages, 'val':paleotmp_interpolateds})

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

plt.plot(sd_interpolated_df.age, scaler.fit_transform(sd_interpolated_df[['val']]))
plt.plot(vd_interpolated_df.age, scaler.fit_transform(vd_interpolated_df[['val']]))
plt.plot(paleotmp_interpolated_df.age, scaler.fit_transform(paleotmp_interpolated_df[['val']]))
plt.plot(netdiv_interpolated_df.age, scaler.fit_transform(netdiv_interpolated_df[['val']]))
plt.plot(dtt_pc1_interpolated_df.age, scaler.fit_transform(dtt_pc1_interpolated_df[['val']]))
plt.plot(dtt_pc2_interpolated_df.age, scaler.fit_transform(dtt_pc2_interpolated_df[['val']]))
plt.plot(dtt_pc3_interpolated_df.age, scaler.fit_transform(dtt_pc3_interpolated_df[['val']]))
plt.show()

plt.scatter(df_interpolated_intersection.age, scaler.fit_transform(df_interpolated_intersection[['stomata_density']]))
plt.scatter(df_interpolated_intersection.age, scaler.fit_transform(df_interpolated_intersection[['vein_density']]))
plt.scatter(df_interpolated_intersection.age, scaler.fit_transform(df_interpolated_intersection[['dtt_pc1']]))
plt.show()

df_interpolated_intersection = pd.DataFrame(dict(
    age=ages, 
    stomata_density=sd_interpolated_df.val,
    vein_density=vd_interpolated_df.val,
    paleotemp=paleotmp_interpolated_df.val,
    net_div_rate=netdiv_interpolated_df.val,
    dtt_pc1=dtt_pc1_interpolated_df.val,
    dtt_pc2=dtt_pc2_interpolated_df.val,
    dtt_pc3=dtt_pc3_interpolated_df.val,
    ))

df_interpolated_intersection = df_interpolated_intersection.dropna()

dfii = df_interpolated_intersection.copy()
dfii.to_csv('./phylo_test/dfii_median.csv', index=False, sep='\t')
# dfii.to_csv('../LepMorphAI/phylo_test/dfii_sparse.csv', index=False, sep='\t')


from scipy.stats import pearsonr, linregress
Xcols = ['stomata_density', 'vein_density', 'paleotemp']
Ycols = [c for c in dfii.columns if c not in Xcols and c != 'age']

for xcol in Xcols:
    for ycol in Ycols:
        #print(xcol, ycol, pearsonr(dfii[xcol], dfii[ycol]))
        lr = linregress([v.sum() for v in dfii[xcol].values], [v.sum() for v in dfii[ycol].values])
        print(xcol, ycol, lr[0], lr[1], lr[2], lr[3], lr[4])




import statsmodels.formula.api as smf

# 加載數據
data = pd.read_csv('../LepMorphAI/phylo_test/dfii_sparse.csv', sep='\t')
data = pd.read_csv('../LepMorphAI/phylo_test/dfii.csv', sep='\t')

plt.plot(data.age, scaler.fit_transform(data[['net_div_rate']]))
plt.plot(data.age, scaler.fit_transform(data[['dtt_pc1']]))
plt.plot(data.age, scaler.fit_transform(data[['dtt_pc2']]))
plt.plot(data.age, scaler.fit_transform(data[['dtt_pc3']]))
plt.plot(data.age, scaler.fit_transform(data[['vein_density']]))
plt.axvline(66)
plt.show()

len(df.family.unique())

# 添加用於分段的虛擬變量
data['age_66_dummy'] = (data['age'] >= 66).astype(str)

# 選擇要分析的 X 和 Y 變量
x_var = 'stomata_density'
y_var = 'net_div_rate'
# y_var = 'dtt_pc1'

# 建立分段回歸模型
formula = f'{y_var} ~ {x_var} * age_66_dummy'
# model = smf.ols(formula, data=data).fit()
model = smf.gls(formula, data=data).fit()

# 顯示模型摘要
print(model.summary())

