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
list(df_ar6clim.index.get_level_values("Variable").unique())

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
nz_scens_index

# %%
# scenarios that are < 47GtCO2e in 2030 are considered to be "immediate action". Those that excced this may fall into the TT30 category.
# This is only really relevant for the C3 and C4 subcategory distinction
imm_scens_index = np.unique(np.where(ghg_emis.loc[:,'2030']<47000)[0])
imm_scens_index

# %%
scen_meta = pd.DataFrame(data=np.zeros((len(ghg_emis), 2), dtype=bool), index=ghg_emis.index, columns=['NZ_GHG', 'IMM'])
scen_meta.iloc[nz_scens_index, 0] = True
scen_meta.iloc[imm_scens_index, 1] = True

# %%
scen_meta

# %%
scen_meta.loc[scen_meta['NZ_GHG']==True]

# %%
scen_meta.loc[scen_meta['IMM']==True]

# %%
ghg_emis.loc[scen_meta['NZ_GHG'], :]

# %%
ghg_emis.loc[scen_meta['IMM'], :]

# %%
pl.plot(ghg_emis.loc[scen_meta['NZ_GHG'], :].values.T);

# %%
pl.plot(ghg_emis.loc[scen_meta['IMM'], :].values.T);

# %% [markdown]
# ## New class proposal
#
# In the below, we first look at how many scenarios would satisfy each class, then we want to take the lowest definition for each scenario. Scenarios could satisfy multiple classes. For example, a scenario that is GW0 would also satisfy GW1 by definition as the latter classification is more "generous" with regards to peak warming and does not impose any different conditions. At the first pass, we don't exclude multiple categories. Later on, we would exclude GW0 scenarios from the GW1 class, and verify that we have precisely 1202 "True" values in our final table.
#
# Following Elmar's suggestion we also define "DEC" based on p50 and p67. We would expect that anything that is DEC at p50 is also at p67. We therefore run "_p50dec" and "_p67dec" versions of the comparison. 
#
# We also use two definitions of warming: smoothed and unsmoothed. This is used to evaluate both peak warming and the rate of decline at the end of century.
#
# We now also distinguish between scenarios that are NZ GHGs before the end of the century and those that are not, and those that are immediate action (IMM) versus those that follow current trends and policies until 2030 (TT30), where the cutoff is 47 GtCO2e emissions in 2030.

# %%
pl.plot(np.arange(2000, 2101), magicc_p50_smoothed.values.T, lw=1, color='b', alpha=0.1);

# %%
pl.plot(np.arange(2000, 2101), magicc_p67_smoothed.values.T, lw=1, color='b', alpha=0.1);

# %%
# GW0a
gw0a_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.5) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) & 
    (scen_meta['NZ_GHG'])
)
np.sum(gw0a_p50dec)

# %%
gw0a_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.5) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) & 
    (scen_meta['NZ_GHG'])
)
np.sum(gw0a_p67dec)

# %%
# GW0b
gw0b_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.5) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) & 
    (~scen_meta['NZ_GHG'])
)
np.sum(gw0b_p50dec)

# %%
gw0b_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.5) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) & 
    (~scen_meta['NZ_GHG'])
)
np.sum(gw0b_p67dec)

# %%
# GW1a
# should be "and not GW0a"; but let's increment one by one for now
gw1a_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.6) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (scen_meta['NZ_GHG'])
)
np.sum(gw1a_p50dec)

# %%
gw1a_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.6) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (scen_meta['NZ_GHG'])
)
np.sum(gw1a_p67dec)

# %%
# GW1b
gw1b_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.6) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG'])
)
np.sum(gw1b_p50dec)

# %%
gw1b_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.6) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG'])
)
np.sum(gw1b_p67dec)

# %%
# GW2-I: not explicitly requiring DEC according to Elmar - only we "expect" it.
# Now with the p50 and p67 definitions we should require it since p67 is a harder condition to meet.
# We are not checking compliance with IMM and NZ. Elmar says we should expect it. However I'm not 100% sure these will all be IMM. 
# most or all should be NZ
gw2i_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p67_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
    # no IMM / NZ condition
)
np.sum(gw2i_p50dec)

# %%
gw2i_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p67_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
    # no IMM / NZ condition
)
np.sum(gw2i_p67dec)

# %%
# GW2-II: not explicitly requiring DEC according to Elmar - only we "expect" it
# Now with the p50 and p67 definitions we should require it since p67 is a harder condition to meet.
# Again not testing IMM+NZ compliance.
gw2ii_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
    # no IMM / NZ condition
)
np.sum(gw2ii_p50dec)

# %%
gw2ii_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
    # no IMM / NZ condition
)
np.sum(gw2ii_p67dec)

# %%
# GW2-IIIa
gw2iiia_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (scen_meta['NZ_GHG'])
)
np.sum(gw2iiia_p50dec)

# %%
gw2iiia_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (scen_meta['NZ_GHG'])
)
np.sum(gw2iiia_p67dec)

# %%
# GW2-IIIb
gw2iiib_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG'])
)
np.sum(gw2iiib_p50dec)

# %%
gw2iiib_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG'])
)
np.sum(gw2iiib_p67dec)

# %%
# GW2-IIIc
gw2iiic_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"])
    # not checking NZ, we assume they are not
)
np.sum(gw2iiic_p50dec)

# %%
gw2iiic_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 1.7) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"])
    # not checking NZ, we assume they are not
)
np.sum(gw2iiic_p67dec)

# %%
# GW3-I
# Elmar says he expects all scenarios to be DEC, but I will put the test in
# not checking NZ: Elmar expects it
# subclass into IMM and TT30
gw3i_imm_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (scen_meta['IMM'])
)
np.sum(gw3i_imm_p50dec)

# %%
gw3i_tt30_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (~scen_meta['IMM'])
)
np.sum(gw3i_tt30_p50dec)

# %%
gw3i_imm_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (scen_meta['IMM'])
)
np.sum(gw3i_imm_p67dec)

# %%
gw3i_tt30_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p50_smoothed["2100"] < 1.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (~scen_meta['IMM'])
)
np.sum(gw3i_tt30_p67dec)

# %%
# GW3-IIa
# requires NZ
# explicitly check DEC
gw3iia_imm_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (scen_meta['NZ_GHG']) &
    (scen_meta['IMM'])
)
np.sum(gw3iia_imm_p50dec)

# %%
gw3iia_tt30_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (scen_meta['NZ_GHG']) &
    (~scen_meta['IMM'])
)
np.sum(gw3iia_tt30_p50dec)

# %%
gw3iia_imm_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (scen_meta['NZ_GHG']) &
    (scen_meta['IMM'])
)
np.sum(gw3iia_imm_p67dec)

# %%
gw3iia_tt30_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (scen_meta['NZ_GHG']) &
    (~scen_meta['IMM'])
)
np.sum(gw3iia_tt30_p67dec)

# %%
# GW3-IIb
gw3iib_imm_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (scen_meta['IMM'])
)
np.sum(gw3iib_imm_p50dec)

# %%
gw3iib_tt30_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (~scen_meta['IMM'])
)
np.sum(gw3iib_tt30_p50dec)

# %%
gw3iib_imm_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (scen_meta['IMM'])
)
np.sum(gw3iib_imm_p67dec)

# %%
gw3iib_tt30_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (~scen_meta['IMM'])
)
np.sum(gw3iib_tt30_p67dec)

# %%
# GW3-IIc
gw3iic_imm_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (scen_meta['IMM'])
)
np.sum(gw3iic_imm_p50dec)

# %%
gw3iic_tt30_p50dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (~scen_meta['IMM'])
)
np.sum(gw3iic_tt30_p50dec)

# %%
gw3iic_imm_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (scen_meta['IMM'])
)
np.sum(gw3iic_imm_p67dec)

# %%
gw3iic_tt30_p67dec = (
    (magicc_p67_smoothed.max(axis=1) < 2.0) &
    (magicc_p67_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG']) &
    (~scen_meta['IMM'])
)
np.sum(gw3iic_tt30_p67dec)

# %%
# GW4-I: not explicitly requiring DEC according to Elmar - only we "expect" it
# Now with the p50 and p67 definitions we should require it since p67 is a harder condition to meet.
# GW4 is not Paris compliant, so does it have to be DEC?
gw4i_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
    # scenarios expected to be NZ; no formal check performed
)
np.sum(gw4i_p50dec)

# %%
gw4i_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 1.7) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
    # scenarios expected to be NZ; no formal check performed
)
np.sum(gw4i_p67dec)

# %%
# GW4-IIa
gw4iia_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 2.0) &
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (scen_meta['NZ_GHG'])
)
np.sum(gw4iia_p50dec)

# %%
gw4iia_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 2.0) &
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (scen_meta['NZ_GHG'])
)
np.sum(gw4iia_p67dec)

# %%
# GW4-IIb
gw4iib_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 2.0) &
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"]) &
    (~scen_meta['NZ_GHG'])
)
np.sum(gw4iib_p50dec)

# %%
gw4iib_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 2.0) &
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"]) &
    (~scen_meta['NZ_GHG'])
)
np.sum(gw4iib_p67dec)

# %%
# GW4-IIc
gw4iic_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 2.0) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"])
    # assuming not NZ as still warming at EoC
)
np.sum(gw4iic_p50dec)

# %%
gw4iic_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.0) & 
    (magicc_p50_smoothed["2100"] < 2.0) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"])
    # assuming not NZ as still warming at EoC
)
np.sum(gw4iic_p67dec)

# %%
# GW5
gw5_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.5) & 
    (magicc_p50_smoothed["2100"] < 2.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
)
np.sum(gw5_p50dec)

# %%
gw5_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.5) & 
    (magicc_p50_smoothed["2100"] < 2.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
)
np.sum(gw5_p67dec)

# %%
# GW5c
# GW5 and GW5c should always be mutually exclusive
gw5c_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.5) & 
    (magicc_p50_smoothed["2100"] < 2.5) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"])
)
np.sum(gw5c_p50dec)

# %%
gw5c_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 2.5) & 
    (magicc_p50_smoothed["2100"] < 2.5) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"])
)
np.sum(gw5c_p67dec)

# %%
# GW6
gw6_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.0) & 
    (magicc_p50_smoothed["2100"] < 3.0) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
)
np.sum(gw6_p50dec)

# %%
# diff of GW6 and GW5 is no scenarios at the p67dec test
gw6_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.0) & 
    (magicc_p50_smoothed["2100"] < 3.0) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
)
np.sum(gw6_p67dec)

# %%
# GW6c
# GW6 and GW6c should always be mutually exclusive
gw6c_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.0) & 
    (magicc_p50_smoothed["2100"] < 3.0) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"])
)
np.sum(gw6c_p50dec)

# %%
gw6c_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.0) & 
    (magicc_p50_smoothed["2100"] < 3.0) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"])
)
np.sum(gw6c_p67dec)

# %%
# GW7. For completeness we'll do the c / no c division, but we expect all to be c in GW7 and GW8
gw7_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.5) & 
    (magicc_p50_smoothed["2100"] < 3.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
)
np.sum(gw7_p50dec)

# %%
gw7_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.5) & 
    (magicc_p50_smoothed["2100"] < 3.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
)
np.sum(gw7_p67dec)

# %%
# GW7c
gw7c_p50dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.5) & 
    (magicc_p50_smoothed["2100"] < 3.5) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"])
)
np.sum(gw7c_p50dec)

# %%
gw7c_p67dec = (
    (magicc_p50_smoothed.max(axis=1) < 3.5) & 
    (magicc_p50_smoothed["2100"] < 3.5) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"])
)
np.sum(gw7c_p67dec)

# %%
# GW8. For completeness we'll do the c / no c division, but we expect all to be c in GW7 and GW8
gw8_p50dec = (
    (magicc_p50_smoothed.max(axis=1) >= 3.5) & 
    (magicc_p50_smoothed["2100"] >= 3.5) & 
    (magicc_p50_smoothed["2100"] < magicc_p50_smoothed["2090"])
)
np.sum(gw8_p50dec)

# %%
gw8_p67dec = (
    (magicc_p50_smoothed.max(axis=1) >= 3.5) & 
    (magicc_p50_smoothed["2100"] >= 3.5) & 
    (magicc_p67_smoothed["2100"] < magicc_p67_smoothed["2090"])
)
np.sum(gw8_p67dec)

# %%
gw8c_p50dec = (
    (magicc_p50_smoothed.max(axis=1) >= 3.5) & 
    (magicc_p50_smoothed["2100"] >= 3.5) & 
    (magicc_p50_smoothed["2100"] >= magicc_p50_smoothed["2090"])
)
np.sum(gw8c_p50dec)

# %%
gw8c_p67dec = (
    (magicc_p50_smoothed.max(axis=1) >= 3.5) & 
    (magicc_p50_smoothed["2100"] >= 3.5) & 
    (magicc_p67_smoothed["2100"] >= magicc_p67_smoothed["2090"])
)
np.sum(gw8c_p67dec)

# %% [markdown]
# Collect into a dataframe, not considering overlaps between categories now

# %%
cats_non_exclusive_p50dec = pd.DataFrame(
    (
        gw0a_p50dec,
        gw0b_p50dec,
        gw1a_p50dec,
        gw1b_p50dec,
        gw2i_p50dec,
        gw2ii_p50dec,
        gw2iiia_p50dec,
        gw2iiib_p50dec,
        gw2iiic_p50dec,
        gw3i_imm_p50dec,
        gw3i_tt30_p50dec,
        gw3iia_imm_p50dec,
        gw3iia_tt30_p50dec,
        gw3iib_imm_p50dec,
        gw3iib_tt30_p50dec,
        gw3iic_imm_p50dec,
        gw3iic_tt30_p50dec,
        gw4i_p50dec,
        gw4iia_p50dec,
        gw4iib_p50dec,
        gw4iic_p50dec,
        gw5_p50dec,
        gw5c_p50dec,
        gw6_p50dec,
        gw6c_p50dec,
        gw7_p50dec,
        gw7c_p50dec,
        gw8_p50dec,
        gw8c_p50dec,
    ),
    index = (
        'GW0a',
        'GW0b',
        'GW1a',
        'GW1b',
        'GW2-I',
        'GW2-II',
        'GW2-IIIa',
        'GW2-IIIb',
        'GW2-IIIc',
        'GW3-I_IMM',
        'GW3-I_TT30',
        'GW3-IIa_IMM',
        'GW3-IIa_TT30',
        'GW3-IIb_IMM',
        'GW3-IIb_TT30',
        'GW3-IIc_IMM',
        'GW3-IIc_TT30',
        'GW4-I',
        'GW4-IIa',
        'GW4-IIb',
        'GW4-IIc',
        'GW5',
        'GW5c',
        'GW6',
        'GW6c',
        'GW7',
        'GW7c',
        'GW8',
        'GW8c'
    )
).T

# %%
cats_non_exclusive_p67dec = pd.DataFrame(
    (
        gw0a_p67dec,
        gw0b_p67dec,
        gw1a_p67dec,
        gw1b_p67dec,
        gw2i_p67dec,
        gw2ii_p67dec,
        gw2iiia_p67dec,
        gw2iiib_p67dec,
        gw2iiic_p67dec,
        gw3i_imm_p67dec,
        gw3i_tt30_p67dec,
        gw3iia_imm_p67dec,
        gw3iia_tt30_p67dec,
        gw3iib_imm_p67dec,
        gw3iib_tt30_p67dec,
        gw3iic_imm_p67dec,
        gw3iic_tt30_p67dec,
        gw4i_p67dec,
        gw4iia_p67dec,
        gw4iib_p67dec,
        gw4iic_p67dec,
        gw5_p67dec,
        gw5c_p67dec,
        gw6_p67dec,
        gw6c_p67dec,
        gw7_p67dec,
        gw7c_p67dec,
        gw8_p67dec,
        gw8c_p67dec,
    ),
    index = (
        'GW0a',
        'GW0b',
        'GW1a',
        'GW1b',
        'GW2-I',
        'GW2-II',
        'GW2-IIIa',
        'GW2-IIIb',
        'GW2-IIIc',
        'GW3-I_IMM',
        'GW3-I_TT30',
        'GW3-IIa_IMM',
        'GW3-IIa_TT30',
        'GW3-IIb_IMM',
        'GW3-IIb_TT30',
        'GW3-IIc_IMM',
        'GW3-IIc_TT30',
        'GW4-I',
        'GW4-IIa',
        'GW4-IIb',
        'GW4-IIc',
        'GW5',
        'GW5c',
        'GW6',
        'GW6c',
        'GW7',
        'GW7c',
        'GW8',
        'GW8c'
    )
).T

# %%
cats_non_exclusive_p50dec

# %%
cats_non_exclusive_p67dec

# %%
cats_non_exclusive_p50dec.sum()

# %%
cats_non_exclusive_p67dec.sum()

# %%
# number of categories satisfied
cats_non_exclusive_p50dec.sum(axis=1)

# %%
cats_non_exclusive_p67dec.sum(axis=1)

# %%
# check: any scenarios not assigned?
(cats_non_exclusive_p50dec.sum(axis=1) == 0).sum()

# %%
(cats_non_exclusive_p67dec.sum(axis=1) == 0).sum()

# %%
# now construct dataframe with exclusivity
# if it's in a lower category, remove it from all higher categories
cats_exclusive_p50dec = cats_non_exclusive_p50dec.copy()
cats_exclusive_p50dec

# %%
cats_exclusive_p67dec = cats_non_exclusive_p67dec.copy()
cats_exclusive_p67dec

# %%
cats = cats_exclusive_p50dec.columns
cats

# %%
cats[0]

# %%
for ic, cat in enumerate(cats[:-1]):
    cats_exclusive_p50dec.loc[cats_exclusive_p50dec[cat]==True, cats[ic+1]:] = False
    cats_exclusive_p67dec.loc[cats_exclusive_p67dec[cat]==True, cats[ic+1]:] = False

# %%
cats_exclusive_p50dec

# %%
cats_exclusive_p67dec

# %%
# scenarios per category
cats_exclusive_p50dec.sum()

# %%
cats_exclusive_p67dec.sum()

# %%
# should be 1202
cats_exclusive_p50dec.sum().sum()

# %%
cats_exclusive_p67dec.sum().sum()

# %%
# should be zero scenarios that do not have one and only one category
(cats_exclusive_p50dec.sum(axis=1) != 1).sum()

# %%
(cats_exclusive_p67dec.sum(axis=1) != 1).sum()

# %%
# make into a two column dataframe
cat_df = pd.DataFrame(index=cats_exclusive_p50dec.index, columns=['Category_p50_DEC', 'Category_p67_DEC'])

# %%
for cat in cats:
    cat_df.loc[cats_exclusive_p50dec[cat], 'Category_p50_DEC'] = cat
    cat_df.loc[cats_exclusive_p67dec[cat], 'Category_p67_DEC'] = cat

# %%
cat_df

# %% [markdown]
# ## make plot

# %%
pl.style.use('../defaults.mplstyle')

# %%
supercats = {
    'GW0': ['GW0a', 'GW0b'],
    'GW1': ['GW1a', 'GW1b'],
    'GW2-I': ['GW2-I'],
    'GW2-II': ['GW2-II'],
    'GW2-III': ['GW2-IIIa', 'GW2-IIIb', 'GW2-IIIc'],
    'GW3-I': ['GW3-I_IMM', 'GW3-I_TT30'],
    'GW3-II': ['GW3-IIa_IMM', 'GW3-IIa_TT30', 'GW3-IIb_IMM', 'GW3-IIb_TT30', 'GW3-IIc_IMM', 'GW3-IIc_TT30'],
    'GW4-I': ['GW4-I'],
    'GW4-II': ['GW4-IIa', 'GW4-IIb', 'GW4-IIc'],
    'GW5': ['GW5', 'GW5c'],
    'GW6': ['GW6', 'GW6c'],
    'GW7+8': ['GW7', 'GW7c', 'GW8', 'GW8c'],
}

# %%
dec_nz_type_p50_DEC = {
    'GW0': ['a', 'b'],
    'GW1': ['a', 'b'],
    'GW2-I': [None],
    'GW2-II': [None],
    'GW2-III': ['a', 'b', 'c'],
    'GW3-I': [None, None],
    'GW3-II': ['a', 'a', 'b', 'b', 'c', 'c'],
    'GW4-I': [None],
    'GW4-II': ['a', 'b', 'c'],
    'GW5': ['b', 'c'],
    'GW6': ['b', 'c'],
    'GW7+8': [None, None, None, None],
}

# %%
dec_nz_type_p67_DEC = {
    'GW0': ['a', 'b'],
    'GW1': ['a', 'b'],
    'GW2-I': [None],
    'GW2-II': [None],
    'GW2-III': ['a', 'b', 'c'],
    'GW3-I': [None, None],
    'GW3-II': ['a', 'a', 'b', 'b', 'c', 'c'],
    'GW4-I': [None],
    'GW4-II': ['a', 'b', 'c'],
    'GW5': ['b', 'c'],
    'GW6': [None, None],
    'GW7+8': [None, None, None, None],
}

# %%
dec_nz_color = {
    'a': 'blue',
    'b': 'red',
    'c': 'green',
    None: 'black',
}

# %%
imm_tt30_type = {
    'GW0': [None, None],
    'GW1': [None, None],
    'GW2-I': [None],
    'GW2-II': [None],
    'GW2-III': [None, None, None],
    'GW3-I': ['IMM', 'TT30'],
    'GW3-II': ['IMM', 'TT30', 'IMM', 'TT30', 'IMM', 'TT30'],
    'GW4-I': [None],
    'GW4-II': [None, None, None],
    'GW5': [None, None],
    'GW6': [None, None],
    'GW7+8': [None, None, None, None],
}

# %%
imm_tt30_ls = {
    None: '-',
    'IMM': '--',
    'TT30': '-.',
}

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
os.makedirs('../plots', exist_ok=True)
fig, ax = pl.subplots(3, 4, figsize=(18/2.54, 18/2.54))
for isc, supercat in enumerate(supercats):
    i = isc//4
    j = isc%4
    #print(i, j, supercat)
    for ic, cat in enumerate(supercats[supercat]):
        ax[i, j].plot(
            np.arange(2000, 2101), 
            magicc_p50_smoothed[cat_df['Category_p50_DEC']==cat].values.T,
            lw=1, 
            color=dec_nz_color[dec_nz_type_p50_DEC[supercat][ic]],
            ls=imm_tt30_ls[imm_tt30_type[supercat][ic]],
            alpha=0.25
        )
    ax[i, j].set_ylim(0.7, 4.0)
    ax[i, j].set_xlim(2000, 2100)
    ax[i, j].text(2005, 3.7, supercat, ha='left', va='baseline')
    ax[i, j].text(2005, 3.4, descriptions_supercats[supercat][0], ha='left', va='baseline')
    ax[i, j].text(2005, 3.1, descriptions_supercats[supercat][1], ha='left', va='baseline')
ax[0, 0].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[0, 1].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[1, 0].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[1, 2].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[2, 0].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[0, 0].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[0, 1].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 0].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 2].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 0].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 1].text(2005, 2.8, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 2].text(2005, 2.8, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 0].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[1, 2].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 0].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 1].text(2005, 2.5, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 2].text(2005, 2.5, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
fig.tight_layout()
pl.savefig('../plots/cats_p50dec.png')

# %%
os.makedirs('../plots', exist_ok=True)
fig, ax = pl.subplots(3, 4, figsize=(18/2.54, 18/2.54))
for isc, supercat in enumerate(supercats):
    i = isc//4
    j = isc%4
    #print(i, j, supercat)
    for ic, cat in enumerate(supercats[supercat]):
        ax[i, j].plot(
            np.arange(2000, 2101), 
            magicc_p50_smoothed[cat_df['Category_p67_DEC']==cat].values.T,
            lw=1, 
            color=dec_nz_color[dec_nz_type_p67_DEC[supercat][ic]],
            ls=imm_tt30_ls[imm_tt30_type[supercat][ic]],
            alpha=0.25
        )
    ax[i, j].set_ylim(0.7, 4.0)
    ax[i, j].set_xlim(2000, 2100)
    ax[i, j].text(2005, 3.7, supercat, ha='left', va='baseline')
    ax[i, j].text(2005, 3.4, descriptions_supercats[supercat][0], ha='left', va='baseline')
    ax[i, j].text(2005, 3.1, descriptions_supercats[supercat][1], ha='left', va='baseline')
ax[0, 0].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[0, 1].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[1, 0].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[1, 2].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[2, 0].text(2005, 2.8, 'a: DEC, NZ', color='blue', ha='left', va='baseline')
ax[0, 0].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[0, 1].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 0].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 2].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 0].text(2005, 2.5, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[2, 1].text(2005, 2.8, 'b: DEC, NoNZ', color='red', ha='left', va='baseline')
ax[1, 0].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[1, 2].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 0].text(2005, 2.2, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
ax[2, 1].text(2005, 2.5, 'c: NoDEC, NoNZ', color='green', ha='left', va='baseline')
fig.tight_layout()
pl.savefig('../plots/cats_p67dec.png')

# %%
