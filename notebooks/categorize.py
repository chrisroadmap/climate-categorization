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
#df_ar6clim.Variable.unique().tolist()

# %%
varp33 = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|33.0th Percentile'
varp50 = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile'
varp67 = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|67.0th Percentile'

# %%
keep_variables = [varp33, varp50, varp67]

# %%
# first step consider only scenarios with a climate assessment (n=1202)
df_vetted = pd.read_csv(os.path.join("..", "data", "models_scenarios_passed_vetting.csv"), index_col=["Model", "Scenario"])
df_vetted

# %%
# data_filtered = []
# for irow, mod_scen in df_vetted.iterrows():
#     model = mod_scen['Model']
#     scenario = mod_scen['Scenario']
#     data_filtered.append(
#         df_ar6clim.loc[
#             (df_ar6clim.index.get_level_values('Model')==model) & 
#             (df_ar6clim.index.get_level_values('Scenario')==scenario) &
#             (df_ar6clim.index.get_level_values("Variable").isin(keep_variables))
#             ]
#         )

# %%
# df_filtered = pd.concat(data_filtered)
# df_filtered

# %%
magicc_p50 = df_ar6clim.loc[df_ar6clim.index.get_level_values("Variable")==varp50]
magicc_p67 = df_ar6clim.loc[df_ar6clim.index.get_level_values("Variable")==varp67]
#magicc_p50.max(axis=1)

# %% [markdown]
# In the below we need to `droplevel` because we want to compare with the same multiindexes, and the variable names differ otherwise

# %%
magicc_p50 = magicc_p50.loc[magicc_p50.index.droplevel(('Region', 'Variable', 'Unit')).isin(df_vetted.index)].droplevel(('Region', 'Variable', 'Unit'))

# %%
magicc_p67 = magicc_p67.loc[magicc_p67.index.droplevel(('Region', 'Variable', 'Unit')).isin(df_vetted.index)].droplevel(('Region', 'Variable', 'Unit'))

# %%
magicc_p50

# %% [markdown]
# ## New class proposal
#
# In the below, we first look at how many scenarios would satisfy each class, then we want to take the lowest definition for each scenario. Scenarios could satisfy multiple classes. For example, a scenario that is GW0 would also satisfy GW1 by definition as the latter classification is more "generous" with regards to peak warming and does not impose any different conditions. At the first pass, we don't exclude multiple categories. Later on, we would exclude GW0 scenarios from the GW1 class, and verify that we have precisely 1202 "True" values in our final table

# %%
# GW0
gw0 = (magicc_p50.max(axis=1) < 1.5) & (magicc_p50["2100"] < 1.5) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw0)

# %%
# GW1
# should be "and not GW0"; but let's increment one by one for now
gw1 = (magicc_p50.max(axis=1) < 1.6) & (magicc_p50["2100"] < 1.5) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw1)

# %%
# GW2-I: not explicitly requiring DEC according to Elmar - only we "expect" it
gw2i = (magicc_p50.max(axis=1) < 1.7) & (magicc_p67["2100"] < 1.5) # & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw2i)

# %%
# GW2-II: not explicitly requiring DEC according to Elmar - only we "expect" it
gw2ii = (magicc_p50.max(axis=1) < 1.7) & (magicc_p50["2100"] < 1.5) # & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw2ii)

# %%
# GW2-III
gw2iii = (magicc_p50.max(axis=1) < 1.7) & (magicc_p50["2100"] < 1.7) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw2iii)

# %%
# GW2-IV: not explicitly requiring DEC according to Elmar - only we "expect" it
gw2iv = (magicc_p67.max(axis=1) < 2.0) & (magicc_p50["2100"] < 1.5) # & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw2iv)

# %%
# GW3
gw3 = (magicc_p67.max(axis=1) < 2.0) & (magicc_p67["2100"] < 2.0) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw3)

# %%
# GW3*
# GW3 and GW3* should always be mutually exclusive here
gw3star = (magicc_p67.max(axis=1) < 2.0) & (magicc_p67["2100"] < 2.0) & (magicc_p50["2100"] >= magicc_p50["2090"])
np.sum(gw3star)

# %%
# GW4-I: not explicitly requiring DEC according to Elmar - only we "expect" it
# GW4 is not Paris compliant, so does it have to be?
gw4i = (magicc_p50.max(axis=1) < 2.0) & (magicc_p50["2100"] < 1.7) #& (magicc_p50["2100"] >= magicc_p50["2090"])
np.sum(gw4i)

# %%
# GW4-II
gw4ii = (magicc_p50.max(axis=1) < 2.0) & (magicc_p50["2100"] < 2.0) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw4ii)

# %%
# GW4-II*
# GW4-II and GW4-II* should always be mutually exclusive
gw4iistar = (magicc_p50.max(axis=1) < 2.0) & (magicc_p50["2100"] < 2.0) & (magicc_p50["2100"] >= magicc_p50["2090"])
np.sum(gw4iistar)

# %%
# GW5
gw5 = (magicc_p50.max(axis=1) < 2.5) & (magicc_p50["2100"] < 2.5) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw5)

# %%
# GW5*
# GW5 and GW5* should always be mutually exclusive
gw5star = (magicc_p50.max(axis=1) < 2.5) & (magicc_p50["2100"] < 2.5) & (magicc_p50["2100"] >= magicc_p50["2090"])
np.sum(gw5star)

# %%
# GW6
# diff of GW6 and GW5 is one scenario, so Elmar's assertion about all GW7 being GW7* might be right
gw6 = (magicc_p50.max(axis=1) < 3.0) & (magicc_p50["2100"] < 3.0) & (magicc_p50["2100"] < magicc_p50["2090"])
np.sum(gw6)

# %%
# GW6*
# GW6 and GW6* should always be mutually exclusive
gw6star = (magicc_p50.max(axis=1) < 3.0) & (magicc_p50["2100"] < 3.0) & (magicc_p50["2100"] >= magicc_p50["2090"])
np.sum(gw6star)

# %%
# GW7*
# Elmar assumes no > 3.0C scenario will be cooling in 2100
gw7star = (magicc_p50.max(axis=1) > 3.0) & (magicc_p50["2100"] > 3.0) #& (magicc_p50["2100"] >= magicc_p50["2090"])
np.sum(gw7star)

# %% [markdown]
# Collect into a dataframe, not considering overlaps between categories now

# %%
cats_non_exclusive = pd.DataFrame(
    (gw0, gw1, gw2i, gw2ii, gw2iii, gw2iv, gw3, gw3star, gw4i, gw4ii, gw4iistar, gw5, gw5star, gw6, gw6star, gw7star),
    index = (
        'GW0',
        'GW1',
        'GW2-I',
        'GW2-II',
        'GW2-III',
        'GW2-IV',
        'GW3',
        'GW3*',
        'GW4-I',
        'GW4-II',
        'GW4-II*',
        'GW5',
        'GW5*',
        'GW6',
        'GW6*',
        'GW7*'
    )
).T

# %%
cats_non_exclusive

# %%
cats_non_exclusive.sum()

# %%
# number of categories satisfied
cats_non_exclusive.sum(axis=1)

# %%
# check: any scenarios not assigned?
(cats_non_exclusive.sum(axis=1) == 0).sum()

# %%
# now construct dataframe with exclusivity
# if it's in a lower category, remove it from all higher categories
cats_exclusive = cats_non_exclusive.copy()
cats_exclusive

# %%
cats = cats_exclusive.columns
cats

# %%
cats[0]

# %%
for ic, cat in enumerate(cats[:-1]):
    cats_exclusive.loc[cats_exclusive[cat]==True, cats[ic+1]:] = False

# %%
cats_exclusive

# %%
# scenarios per category
cats_exclusive.sum()

# %%
# should be 1202
cats_exclusive.sum().sum()

# %%
# should be zero scenarios that do not have one and only one category
(cats_exclusive.sum(axis=1) != 1).sum()

# %%
# let's now condense into a one-column dataframe
cat_df = pd.DataFrame(index=cats_exclusive.index, columns=['Category'])

# %%
cat_df

# %%
# for idx, data in cat_df.iterrows():
#     model = idx[0]
#     scenario = idx[1]
#     print(cats_exclusive.loc[(model, scenario)])

# %%
for cat in cats:
    cat_df.loc[cats_exclusive[cat], 'Category'] = cat

# %%
cat_df

# %% [markdown]
# ## make plot

# %%
pl.style.use('../defaults.mplstyle')

# %%
descriptions = {
    'GW0': ('PW$_{50}$<1.5°C', 'EoCW$_{50}$<1.5°C', 'DEC'),
    'GW1': ('PW$_{50}$<1.6°C', 'EoCW$_{50}$<1.5°C', 'DEC'),
    'GW2-I': ('PW$_{50}$<1.7°C', 'EoCW$_{67}$<1.5°C', 'DEC (expected)'),
    'GW2-II': ('PW$_{50}$<1.7°C', 'EoCW$_{50}$<1.5°C', 'DEC (expected)'),
    'GW2-III': ('PW$_{50}$<1.7°C', 'EoCW$_{50}$<1.7°C', 'DEC'),
    'GW2-IV': ('PW$_{67}$<2.0°C', 'EoCW$_{50}$<1.5°C', 'DEC (expected)'),
    'GW3': ('PW$_{67}$<2.0°C', 'EoCW$_{67}$<2.0°C', 'DEC'),
    'GW3*': ('PW$_{67}$<2.0°C', 'EoCW$_{67}$<2.0°C', 'NoDEC'),
    'GW4-I': ('PW$_{50}$<2.0°C', 'EoCW$_{50}$<1.7°C', 'DEC (expected)'),
    'GW4-II': ('PW$_{50}$<2.0°C', 'EoCW$_{50}$<2.0°C', 'DEC'),
    'GW4-II*': ('PW$_{50}$<2.0°C', 'EoCW$_{50}$<2.0°C', 'NoDEC'),
    'GW5': ('PW$_{50}$<2.5°C', 'EoCW$_{50}$<2.5°C', 'DEC'),
    'GW5*': ('PW$_{50}$<2.5°C', 'EoCW$_{50}$<2.5°C', 'NoDEC'),
    'GW6': ('PW$_{50}$<3.0°C', 'EoCW$_{50}$<3.0°C', 'DEC'),
    'GW6*': ('PW$_{50}$<3.0°C', 'EoCW$_{50}$<3.0°C', 'NoDEC'),
    'GW7*': ('PW$_{50}$>=3.0°C', 'EoCW$_{50}$>=3.0°C', 'NoDEC (expected)'),
}

# %%
os.makedirs('../plots', exist_ok=True)
fig, ax = pl.subplots(4, 4, figsize=(18/2.54, 18/2.54))
for ic, cat in enumerate(cats):
    i = ic//4
    j = ic%4
    ax[i, j].plot(np.arange(1995, 2101), magicc_p50[cat_df['Category']==cat].values.T, lw=1, color='b', alpha=0.5)
    ax[i, j].set_ylim(0.7, 4.0)
    ax[i, j].set_xlim(2000, 2100)
    ax[i, j].text(2005, 3.7, cat, ha='left', va='baseline')
    ax[i, j].text(2005, 3.4, descriptions[cat][0], ha='left', va='baseline')
    ax[i, j].text(2005, 3.1, descriptions[cat][1], ha='left', va='baseline')
    ax[i, j].text(2005, 2.8, descriptions[cat][2], ha='left', va='baseline')
    ax[i, j].text(2005, 2.5, f'n={cats_exclusive.sum()[cat]}', ha='left', va='baseline')
fig.tight_layout()
pl.savefig('../plots/cats.png')

# %%
