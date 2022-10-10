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
avi_file = "TrackingData/B_DF_15s_7_5fps_3.avi"
# %%
# pims.Video henter avi fil, og bruker gray() til gray channel
frames = gray(pims.Video(avi_file))
f = tp.batch(frames[:-1], 21, invert=True)
# %%
tp.quiet()
t=tp.link(f, 5, memory=5)
# %% 
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count')
# %%
t1 = tp.filter_stubs(t,25)
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())
# %%
plt.figure()
tp.mass_size(t1.groupby('particle').mean())
#%%
t2 = t1[((t1['mass'] < 30000) & (t1['size'] < 40) & (t1['ecc'] > 0.00000003))]
plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0])
# %%
plt.figure()
tp.plot_traj(t2)
# %%
d = tp.compute_drift(t2)
d.plot()
plt.show()
# %%
tm = tp.subtract_drift(t2.copy(), d)
ax = tp.plot_traj(tm)
plt.show()
# %%
em = tp.emsd(tm,125/48., 7.5)
# %%
fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel = 'lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set(ylim=(1e-2,10))
# %%
plt.figure()
plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
plt.xlabel('lag time $t$')
tp.utils.fit_powerlaw(em)





