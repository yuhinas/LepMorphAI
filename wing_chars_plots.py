import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text
mpl.rcParams['pdf.fonttype'] = 42

fwchars = pd.read_csv(f'./wing_characters/fam_fw_chars_20230518.csv', sep='\t')
fwchars['wing'] = 'fw'

hwchars = pd.read_csv(f'./wing_characters/fam_hw_chars_20230518.csv', sep='\t')
hwchars['wing'] = 'hw'


wchars = pd.concat([fwchars, hwchars]).reset_index(drop=True)
wchars['group'] = 'primative'
wchars.loc[wchars['6groups'].isin([2,4,5]), 'group'] = 'more_recent'

# wchars[wchars.family=='Nymphalidae']['distinctive_from_wing_base']

# wchars.columns

# wchars_colors = wchars[['color_saturation', 'color_brightness', 'color_richness', 'color_evenness', 'distinctive_from_wing_base', 'group']]

# mu_ = wchars_colors.groupby('group').mean()
# std_ = wchars_colors.groupby('group').std()

# std_ / mu_

# q1 = wchars_colors.groupby('group').quantile(.25)
# q2 = wchars_colors.groupby('group').quantile(.5)
# q3 = wchars_colors.groupby('group').quantile(.75)

# (q3 - q1) / (q3 + q1)

color_cols = ['color_saturation', 'color_brightness', 'color_richness', 'color_evenness', 'distinctive_from_wing_base']
#color_cols = ['color_saturation', 'color_richness']
#color_cols = ['aspect_ratio', 'secondMoA']

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

fam_after_660Ma_group = ['Brahmaeidae', 'Eupterotidae', 'Noctuidae', 'Euteliidae', 'Nolidae', 'Lycaenidae', 'Riodinidae']
fam_top20 = wchars.sort_values('Goldstein2017').iloc[-20:].family.values
edgecolors = ['red' if f in fam_after_660Ma_group else 'gray' for f in wchars.family]
linewidths = [3 if f in fam_after_660Ma_group else 1 for f in wchars.family]


subdf_primitive_fw = wchars.query("group == 'primative' and wing == 'fw'")
subdf_primitive_hw = wchars.query("group == 'primative' and wing == 'hw'")
subdf_recent_fw = wchars.query("group == 'more_recent' and wing == 'fw'")
subdf_recent_hw = wchars.query("group == 'more_recent' and wing == 'hw'")

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.rcParams['figure.figsize'] = [5, 5]
g1p = 1
g2p = 2
offset = .1
for i in range(len(color_cols)):
    char1 = color_cols[i]
    #plt.boxplot([[subdf_primitive_fw[char1].values, subdf_recent_fw[char1].values], [subdf_primitive_hw[char1].values, subdf_recent_fw[char1].values]])
    bpp = plt.boxplot([subdf_primitive_fw[char1].values, subdf_primitive_hw[char1].values], positions=[g1p-offset, g2p-offset], labels=['primitive', 'primitive'], flierprops={'markersize': 3, 'markerfacecolor': '#2C7BB6', 'markeredgecolor': 'blue', 'linewidth': 1}, widths=.12)
    bpr = plt.boxplot([subdf_recent_fw[char1].values, subdf_recent_hw[char1].values], positions=[g1p+offset, g2p+offset], labels=['more recent', 'more recent'], flierprops={'markersize': 3, 'markerfacecolor': '#D7191C', 'markeredgecolor': 'red', 'linewidth': 1}, widths=.12)
    set_box_color(bpp, color='#2C7BB6')
    set_box_color(bpr, color='#D7191C')
    plt.xticks([1, 2], ['fore wing', 'hind wing'])
    plt.title(char1)
    plt.tight_layout()
    plt.legend([bpp["fliers"][0], bpr["fliers"][0]], ['primitive', 'more recent'], loc='upper right', fontsize=7)
    plt.savefig(f'./wing_characters/aw/boxplot-{char1}_20230518.pdf')
    plt.savefig(f'./wing_characters/aw/boxplot-{char1}_20230518.png')
    plt.close()


wing = 'hw'
plt.rcParams['figure.figsize'] = [15, 15]
for i in range(len(color_cols)):
    char1 = color_cols[i]
    # plt.boxplot([wchars.loc[wchars['group']=='primative', char1], wchars.loc[wchars['group']=='more_recent', char1]])
    # plt.xticks([1, 2], ['primative', 'more_recent'])
    # plt.title(char1)
    # plt.savefig(f'{wing}/boxplot-{char1}.pdf')
    # plt.savefig(f'{wing}/boxplot-{char1}.png')
    # plt.close()

    for j in range(i+1, len(color_cols)):
        print(i, j)
        char2 = color_cols[j]
        plt.scatter(wchars.query('wing==@wing')[char1], wchars.query('wing==@wing')[char2], c=[get_fam_color(f) for f in wchars.query('wing==@wing').family], s=(np.sqrt(np.log(wchars.query('wing==@wing').Goldstein2017.values) * 1)).astype(int)*100, edgecolor='gray', linewidth=1)

        texts = []
        for _, row_ in wchars.iterrows():
            if row_.family in fam_top20: #fam_after_660Ma_group:
                texts.append(plt.text(row_[char1], row_[char2], row_.family, fontsize=12))
                #texts.append(plt.text(row.nd2ndMoA_root_on_right, row.aspect_ratio, row.img_file[:-4], fontsize=ax_fontsize))

        adjust_text(texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='lightgray', lw=1))

        plt.title(wing.upper())
        plt.xlabel(char1)
        plt.ylabel(char2)
        plt.show()
        # plt.savefig(f'{wing}/v20230510-comb2-{char1}-{char2}-fam_colored.pdf')
        # plt.savefig(f'{wing}/v20230510-comb2-{char1}-{char2}-fam_colored.png')
        #plt.savefig(f'{wing}/comb2-{char1}-{char2}.pdf')
        #plt.savefig(f'{wing}/comb2-{char1}-{char2}.png')
        # plt.close()

