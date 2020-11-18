# Nicholas M. Rathmann, NBI, UCPH, 2020.
# Sponsered by Villum Fonden as part of the project "IceFlow".

import sys
import pandas
import matplotlib.pyplot as plt

fname = sys.argv[1] # commandline argument #1
df = pandas.read_csv(fname, sep=',')

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12,7))
f.suptitle(fname + ' (%i grains)'%(len(df['grain_number'])))
bins = 40

ax1.hist(df['equivalent_diameter'], bins=bins, color='k')
ax1.set_xlabel('Equiv. diameter [px]')

ax2.hist(df['area'], bins=bins, color='0.4')
ax2.set_xlabel('Area [px^2]')

ax4.hist(df['major_axis_length'], bins=bins, color='#1f78b4')
ax4.set_xlabel('Major axis [px]')

ax5.hist(df['eccentricity'], bins=bins, color='#6a3d9a')
ax5.set_xlabel('Eccentricity')

ax3.hist(df['perimeter'], bins=bins, color='#33a02c')
ax3.set_xlabel('Perimeter [px]')

ax6.hist(df['orientation'], bins=bins, color='#b15928')
ax6.set_xlabel('Orientation [deg.]')

plt.tight_layout()
plt.savefig(fname[:-4]+'_grainstats.png')