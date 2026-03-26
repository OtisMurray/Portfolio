import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

params = {'font.family': 'sans-serif',
          'font.serif': 'Calibri',
          'font.style': 'normal',
          'font.weight': 'bold',
          'font.size': 10,
          }
rcParams.update(params)

input_directory = r"E:\GHAP\MB.xlsx"
output_directory = r"E:\GHAP"

df = pd.read_excel(input_directory, sheet_name=0)
dataset = df.values

target_data = dataset[:, 1]
fig = plt.figure(dpi=200, figsize=(3, 3))
ax1 = plt.subplot(1, 1, 1)
plt.rcParams["patch.force_edgecolor"] = True

sns.distplot(target_data, hist=False, kde=False, color="#000000",
             bins=10,
             hist_kws={'histtype': 'bar', 'linewidth': 2, 'edgecolor': 'gray', 'fill': False, 'alpha': 0.3},
             fit=stats.norm,
             fit_kws={'color': '#000000', 'linestyle': '-', 'linewidth': 2})

plt.axvline(0, label='mean', linestyle='--', color='#000000', linewidth=2)
ax1.set_ylim(0, 0.0008)
ax1.set_xlim(-6000, 6000)

ax1.set_xticks([-6000, -4000, -2000, -0, 2000, 4000, 6000])
ax1.set_xticklabels([-6, -4, -2, -0, 2, 4, 6], color='#000000')

ax1.set_yticks([0, 0.0002, 0.0004, 0.0006, 0.0008], minor=False)
ax1.set_yticklabels([0, 2, 4, 6, 8], color='#000000')

ax1.tick_params(bottom='on', left='on', which='major', direction='out', width=2.0, length=4.0, color='#000000')
ax1.tick_params(bottom='on', left='on', which='minor', direction='out', width=2.0, length=3.0, color='#000000')

ax1.set_xlabel('Difference (thousand)', fontweight='bold', labelpad=4, color='#000000')
ax1.set_ylabel('Probability Density ' + r'$(10^{-4})$', fontweight='bold', labelpad=4, color='#000000')

ax1.spines['bottom'].set_linewidth('2.0')
ax1.spines['left'].set_linewidth('2.0')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#000000')
ax1.spines['bottom'].set_color('#000000')

result_name = output_directory + r'/MB_PreCOVID_PDF.jpg'
print(result_name)
plt.savefig(result_name, dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.show()