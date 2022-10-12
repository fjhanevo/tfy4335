# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
from __future__ import division, unicode_literals, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from pandas import DataFrame, Series
import pims
import trackpy as tp
# %%
# henter ut gray channel
@pims.pipeline
def gray(image):
    return image[:, :, 1]
# %%
# lagrer avi fil i variabel
avi_file = "TrackingData/A_DF_40x_15s_5fps_1_cropped-1.avi"
# %%
# pims.Video henter avi fil, og bruker gray() til gray channel
frames = gray(pims.open(avi_file))

# %%
f = tp.locate(frames[0], 43)
tp.annotate(f, frames[0])

# %%
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# %%
f = tp.locate(frames[0], 43, minmass=2000)
tp.annotate(f, frames[0])

# %%
f = tp.batch(frames[:], 43, minmass = 2000)
# %%
t = tp.link(f, 5, memory=3)
t.head()
# %%
t1 = tp.filter_stubs(t, 25)
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())
# %%
plt.figure()
tp.mass_size(t1.groupby('particle').mean())
# %%
t2 = t1[((t1['mass'] > 50) & (t1['size'] < 10) & (t1['ecc'] < 3))]
plt.figure()
tp.annotate(t2[t2['frame']==0], frames[0])
# %% 
plt.figure()
tp.plot_traj(t2)
# %%
d = tp.compute_drift(t2)
d.plot()
plt.show()
# %%
# im = tp.imsd(tm, 100/285., 5)
# fig, ax = plt.subplots()
# ax.plot(im.index, im, 'k-', alpha=0.1)
# ax.set_xscale('log')
# ax.set_yscale('log')
# %%
em = tp.emsd(tm, 1, 24)
fig, ax = plt.subplots()
ax.plot(em.index,em,'o')
ax.set_xscale('log')
ax.set_yscale('log')
# %%
plt.figure()
tp.utils.fit_powerlaw(em)

