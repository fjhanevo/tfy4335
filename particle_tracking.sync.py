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
avi_file = "Continuous-A_DF_1.avi"
# %%
# pims.Video henter avi fil, og bruker gray() til gray channel
frames = gray(pims.Video(avi_file))
# %%
# frames[0] henter ut en enkelt frame fra filen
f = tp.locate(frames[0], 11, invert=True)
# %% 
# viser første radene av data
f.head()
# %%
# lager supblots for histogrammene
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count')
# %%
f = tp.locate(frames[0], 11, invert=True, minmass=20)
tp.annotate(f, frames[0])
