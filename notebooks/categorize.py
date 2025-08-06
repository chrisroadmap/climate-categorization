# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

# %%
# the AR6 climate diagnostics
# this is a huge file (3GB) and not in the public domain
# users will want to download this file and change this path
datadir = os.path.join("C:\\", "Users", "CS000052", "data", "is-c1-dead", "input")
filename = "AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv"

# %%
datadir

# %%
df_ar6clim = pd.read_csv(os.path.join(datadir, filename), index_col=["Model", "Scenario", "Region", "Variable", "Unit"])
df_ar6clim

# %%
#list(df_ar6clim.index.get_level_values("Variable").unique())

# %%
varp50 = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile'
varp67 = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|67.0th Percentile'
ghg_emis = 'AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)'

# %%
keep_variables = [varp50, varp67, ghg_emis]

# %%
# first step consider only scenarios with a climate assessment (n=1202)
df_vetted = pd.read_csv(os.path.join("..", "data", "models_scenarios_passed_vetting.csv"), index_col=["Model", "Scenario"])
df_vetted

# %%
# drop pre-2000 which are NaN and cause problems with the savgol filter
magicc_p50 = df_ar6clim.loc[df_ar6clim.index.get_level_values("Variable")==varp50, '2000':'2100']
magicc_p67 = df_ar6clim.loc[df_ar6clim.index.get_level_values("Variable")==varp67, '2000':'2100']
ghg_emis = df_ar6clim.loc[df_ar6clim.index.get_level_values("Variable")==ghg_emis, '2000':'2100']
#magicc_p50.max(axis=1)

# %% [markdown]
# In the below we need to `droplevel` because we want to compare with the same multiindexes, and the variable names differ otherwise

# %%
magicc_p50 = magicc_p50.loc[magicc_p50.index.droplevel(('Region', 'Variable', 'Unit')).isin(df_vetted.index)].droplevel(('Region', 'Variable', 'Unit'))

# %%
magicc_p67 = magicc_p67.loc[magicc_p67.index.droplevel(('Region', 'Variable', 'Unit')).isin(df_vetted.index)].droplevel(('Region', 'Variable', 'Unit'))

# %%
# technically we should be looking at the statistics of the smoothed, not the smoothing of the statistics, but we don't have
# access to the raw model output. GHG emissions should be smooth since IAMs are 5- or 10-year timesteps
magicc_p50_smoothed = magicc_p50.apply(lambda x: savgol_filter(x, 11, 1, mode='interp'), axis=1, result_type='broadcast')
magicc_p67_smoothed = magicc_p67.apply(lambda x: savgol_filter(x, 11, 1, mode='interp'), axis=1, result_type='broadcast')

# %%
magicc_p50_smoothed

# %%
magicc_p67_smoothed

# %%
ghg_emis = ghg_emis.loc[ghg_emis.index.droplevel(('Region', 'Variable', 'Unit')).isin(df_vetted.index)].droplevel(('Region', 'Variable', 'Unit'))

# %%
# indices of scenarios that achieve NZ GHGs before 2100
nz_scens_index = np.unique(np.where(np.diff(np.sign(ghg_emis))<0)[0])
# nz_scens_index

# %%
#not_nz_scens_index = np.arange(1202, dtype=int)
not_nz_scens_index = np.array([i for i in np.arange(1202, dtype=int) if i not in nz_scens_index])
#not_nz_scens_index

# %%
# # scenarios that are < 47GtCO2e in 2030 are considered to be "immediate action". Those that excced this may fall into the TT30 category.
# # This is only relevant for the C3 subcategory
# imm_scens_index = np.unique(np.where(ghg_emis.loc[:,'2030']<47000)[0])
# imm_scens_index

# %%
meta_columns = [
    'Category_DEC_p50',
    'Category_DEC_p67',
    'PW_p50',
    'PW_p67',
    '2100_p50',
    '2100_p67',
    '2090_p50',
    '2090_p67',
    'DEC_p50',
    'DEC_p67',
    'NZ_GHG', 
    'IMM_TT30', 
]

# %%
scen_meta = pd.DataFrame(
    #data=np.zeros((len(ghg_emis), len(meta_columns)), dtype=bool), 
    index=ghg_emis.index, 
    columns=meta_columns,
)
# Fill in emissions-based meta now
scen_meta.iloc[not_nz_scens_index, scen_meta.columns.get_loc('NZ_GHG')] = False
scen_meta.iloc[nz_scens_index, scen_meta.columns.get_loc('NZ_GHG')] = True
#scen_meta.iloc[imm_scens_index, scen_meta.columns.get_loc('IMM_TT30')] = 'IMM'

# %%
scen_meta

# %%
ghg_emis.loc[scen_meta['NZ_GHG'], :]

# %%
pl.plot(ghg_emis.loc[scen_meta['NZ_GHG'], :].values.T);

# %% [markdown]
# ## New class proposal
#
# In the below, we first look at how many scenarios would satisfy each class, then we want to take the lowest definition for each scenario. Scenarios could satisfy multiple classes. For example, a scenario that is GW0 would also satisfy GW1 by definition as the latter classification is more "generous" with regards to peak warming and does not impose any different conditions. At the first pass, we don't exclude multiple categories. Later on, we would exclude GW0 scenarios from the GW1 class, and verify that we have precisely 1202 "True" values in our final table.
#
# Following Elmar's suggestion we also define "DEC" based on p50 and p67. We would expect that anything that is DEC at p50 is also at p67. We therefore run "_p50dec" and "_p67dec" versions of the comparison. 
#
# We also use a smoothed definition of warming for both peak warming and the rate of decline at the end of century. This is because we want to isolate the effect of the solar cycle.
#
# We now also distinguish between scenarios that are NZ GHGs before the end of the century and those that are not, and those that are immediate action (IMM) versus those that follow current trends and policies until 2030 (TT30), where the cutoff is 47 GtCO2e emissions in 2030.

# %%
pl.plot(np.arange(2000, 2101), magicc_p50_smoothed.values.T, lw=1, color='b', alpha=0.1);

# %%
pl.plot(np.arange(2000, 2101), magicc_p67_smoothed.values.T, lw=1, color='b', alpha=0.1);

# %%
magicc_p50_smoothed.max(axis=1)

# %%
scen_meta.loc[:, 'PW_p50'] = magicc_p50_smoothed.max(axis=1)
scen_meta.loc[:, 'PW_p67'] = magicc_p67_smoothed.max(axis=1)
scen_meta.loc[:, '2100_p50'] = magicc_p50_smoothed["2100"]
scen_meta.loc[:, '2100_p67'] = magicc_p67_smoothed["2100"]
scen_meta.loc[:, '2090_p50'] = magicc_p50_smoothed["2090"]
scen_meta.loc[:, '2090_p67'] = magicc_p67_smoothed["2090"]
scen_meta.loc[:, 'DEC_p50'] = scen_meta.loc[:, '2100_p50'] < scen_meta.loc[:, '2090_p50']
scen_meta.loc[:, 'DEC_p67'] = scen_meta.loc[:, '2100_p67'] < scen_meta.loc[:, '2090_p67']
scen_meta

# %%
for dec_level in [50, 67]:
    scen_meta.loc[(scen_meta['PW_p50']<1.5) & (scen_meta['2100_p50']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']), f'Category_DEC_p{dec_level}'] = 'GW0a'
    scen_meta.loc[(scen_meta['PW_p50']<1.5) & (scen_meta['2100_p50']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']==False), f'Category_DEC_p{dec_level}'] = 'GW0b'
    
    scen_meta.loc[(scen_meta['PW_p50']<1.6) & (scen_meta['2100_p50']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW1a'
    scen_meta.loc[(scen_meta['PW_p50']<1.6) & (scen_meta['2100_p50']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW1b'
    
    scen_meta.loc[(scen_meta['PW_p50']<1.7) & (scen_meta['2100_p67']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW2-I'
    scen_meta.loc[(scen_meta['PW_p50']<1.7) & (scen_meta['2100_p50']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW2-II'
    scen_meta.loc[(scen_meta['PW_p50']<1.7) & (scen_meta['2100_p50']<1.7) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW2-IIIa'
    scen_meta.loc[(scen_meta['PW_p50']<1.7) & (scen_meta['2100_p50']<1.7) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW2-IIIb'
    scen_meta.loc[(scen_meta['PW_p50']<1.7) & (scen_meta['2100_p50']<1.7) & (scen_meta[f'DEC_p{dec_level}']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW2-IIIc'
    
    scen_meta.loc[(scen_meta['PW_p67']<2.0) & (scen_meta['2100_p50']<1.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW3-I'
    scen_meta.loc[(scen_meta['PW_p67']<2.0) & (scen_meta['2100_p67']<2.0) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW3-IIa'
    scen_meta.loc[(scen_meta['PW_p67']<2.0) & (scen_meta['2100_p67']<2.0) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW3-IIb'
    scen_meta.loc[(scen_meta['PW_p67']<2.0) & (scen_meta['2100_p67']<2.0) & (scen_meta[f'DEC_p{dec_level}']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW3-IIc'
    
    scen_meta.loc[(scen_meta['PW_p50']<2.0) & (scen_meta['2100_p50']<1.7) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW4-I'
    scen_meta.loc[(scen_meta['PW_p50']<2.0) & (scen_meta['2100_p50']<2.0) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW4-IIa'
    scen_meta.loc[(scen_meta['PW_p50']<2.0) & (scen_meta['2100_p50']<2.0) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW4-IIb'
    scen_meta.loc[(scen_meta['PW_p50']<2.0) & (scen_meta['2100_p50']<2.0) & (scen_meta[f'DEC_p{dec_level}']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW4-IIc'
    
    scen_meta.loc[(scen_meta['PW_p50']<2.5) & (scen_meta['2100_p50']<2.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW5a'
    scen_meta.loc[(scen_meta['PW_p50']<2.5) & (scen_meta['2100_p50']<2.5) & (scen_meta[f'DEC_p{dec_level}']) & (scen_meta['NZ_GHG']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW5b'
    scen_meta.loc[(scen_meta['PW_p50']<2.5) & (scen_meta['2100_p50']<2.5) & (scen_meta[f'DEC_p{dec_level}']==False) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW5c'
    
    scen_meta.loc[(scen_meta['PW_p50']<3.0) & (scen_meta['2100_p50']<3.0) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW6'
    scen_meta.loc[(scen_meta['PW_p50']<3.5) & (scen_meta['2100_p50']<3.5) & (scen_meta[f'Category_DEC_p{dec_level}'].isna()), f'Category_DEC_p{dec_level}'] = 'GW7'
    scen_meta.loc[(scen_meta['PW_p50']>=3.5) & (scen_meta['2100_p50']>=3.5), f'Category_DEC_p{dec_level}'] = 'GW8'

# %%
scen_meta

# %%
# Next separate the GW3 category into IMM and TT30. Not relevant for other categories. 
scen_meta.loc[scen_meta['Category_DEC_p50'].str.startswith('GW3')].index

# %%
test_2030 = ghg_emis.loc[scen_meta.loc[scen_meta['Category_DEC_p50'].str.startswith('GW3')].index, '2030'] < 47000
for idx, lt47gt in test_2030.items():
    if lt47gt:
        scen_meta.loc[idx, 'IMM_TT30'] = 'IMM'
    else:
        scen_meta.loc[idx, 'IMM_TT30'] = 'TT30'
#), 'IMM_TT30']

# %%
# finally rename the DEC_p50
scen_meta.rename(columns={"Category_DEC_p50": "Category"}, inplace=True)
scen_meta

# %%
# scenarios per category
supercats = {
    'GW0': ['GW0a', 'GW0b'],
    'GW1': ['GW1a', 'GW1b'],
    'GW2-I': ['GW2-I'],
    'GW2-II': ['GW2-II'],
    'GW2-III': ['GW2-IIIa', 'GW2-IIIb', 'GW2-IIIc'],
    'GW3-I': ['GW3-I'],
    'GW3-II': ['GW3-IIa', 'GW3-IIb', 'GW3-IIc'],
    'GW4-I': ['GW4-I'],
    'GW4-II': ['GW4-IIa', 'GW4-IIb', 'GW4-IIc'],
    'GW5': ['GW5a', 'GW5b', 'GW5c'],
    'GW6': ['GW6'],
    'GW7+8': ['GW7', 'GW8'],
}

# %%
cats = []
for cat in supercats.values():
    for icat in cat:
        cats.append(icat)
cats

# %%
for cat in cats:
    print((scen_meta['Category']==cat).sum())

# %% [markdown]
# ## make plot

# %%
pl.style.use('../defaults.mplstyle')

# %%
descriptions_supercats = {
    'GW0': ('PW$_{50}$<1.5°C', 'EoCW$_{50}$<1.5°C'),
    'GW1': ('PW$_{50}$<1.6°C', 'EoCW$_{50}$<1.5°C'),
    'GW2-I': ('PW$_{50}$<1.7°C', 'EoCW$_{67}$<1.5°C'),
    'GW2-II': ('PW$_{50}$<1.7°C', 'EoCW$_{50}$<1.5°C'),
    'GW2-III': ('PW$_{50}$<1.7°C', 'EoCW$_{50}$<1.7°C'),
    'GW3-I': ('PW$_{67}$<2.0°C', 'EoCW$_{50}$<1.5°C'),
    'GW3-II': ('PW$_{67}$<2.0°C', 'EoCW$_{67}$<2.0°C'),
    'GW4-I': ('PW$_{50}$<2.0°C', 'EoCW$_{50}$<1.7°C'),
    'GW4-II': ('PW$_{50}$<2.0°C', 'EoCW$_{50}$<2.0°C'),
    'GW5': ('PW$_{50}$<2.5°C', 'EoCW$_{50}$<2.5°C'),
    'GW6': ('PW$_{50}$<3.0°C', 'EoCW$_{50}$<3.0°C'),
    'GW7+8': ('PW$_{50}$>=3.0°C', 'EoCW$_{50}$>=3.0°C'),
}

# %%
dec_nz_type = {
    'GW0': ['a', 'b'],
    'GW1': ['a', 'b'],
    'GW2-I': [None],
    'GW2-II': [None],
    'GW2-III': ['a', 'b', 'c'],
    'GW3-I': [None, None],
    'GW3-II': ['a', 'a', 'b', 'b', 'c', 'c'],
    'GW4-I': [None],
    'GW4-II': ['a', 'b', 'c'],
    'GW5': ['a', 'b', 'c'],
    'GW6': [None],
    'GW7+8': [None, None],
}

dec_nz_color = {
    'a': 'blue',
    'b': 'red',
    'c': 'green',
    '7': 'orange',
    '8': 'purple',
    None: 'black',
}

# %%
os.makedirs('../plots', exist_ok=True)
fig, ax = pl.subplots(3, 4, figsize=(18/2.54, 18/2.54))
for isc, supercat in enumerate(supercats):
    i = isc//4
    j = isc%4
    #print(i, j, supercat)
    n = 0
    for ic, cat in enumerate(supercats[supercat]):
        ax[i, j].plot(
            np.arange(2000, 2101), 
            magicc_p50_smoothed[scen_meta['Category']==cat].values.T,
            lw=1, 
            color=dec_nz_color[dec_nz_type[supercat][ic]],
            alpha=0.25
        )
        n = n + (scen_meta['Category']==cat).sum()
    ax[i, j].set_ylim(0.7, 4.0)
    ax[i, j].set_xlim(2000, 2100)
    ax[i, j].text(2005, 3.7, supercat, ha='left', va='baseline')
    ax[i, j].text(2005, 3.45, descriptions_supercats[supercat][0], ha='left', va='baseline')
    ax[i, j].text(2005, 3.2, descriptions_supercats[supercat][1], ha='left', va='baseline')
    ax[i, j].text(2005, 2.95, f'n = {n}', va='baseline')
ax[0, 0].text(2005, 2.7, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[0, 1].text(2005, 2.7, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[1, 0].text(2005, 2.7, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[1, 2].text(2005, 2.7, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[2, 0].text(2005, 2.7, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[2, 1].text(2005, 2.7, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[0, 0].text(2005, 2.45, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[0, 1].text(2005, 2.45, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 0].text(2005, 2.45, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 2].text(2005, 2.45, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 0].text(2005, 2.45, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 1].text(2005, 2.45, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 0].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[1, 2].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 0].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 1].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
fig.tight_layout()
pl.savefig('../plots/cats_p50dec.png')

# %%
# export metadata
os.makedirs('../output')
scen_meta.to_csv('../output/meta.csv')
