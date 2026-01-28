import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.weight']='bold'
plt.rcParams['font.size'] = 12
# plt.rcParams['image.cmap'] = 'gist_heat' 
plt.rcParams['image.cmap'] = 'jet' 

from pathlib import Path

# --- Path Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "input" / "Noise_file"
OUTPUT_DIR = PROJECT_ROOT / "output" / "noise_fsp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

freq = []
tmp = []
amp = []
# total = list()
graph_color = ['k', 'b', 'r', 'g', 'c']

for i in range(0, 5):
    fpath = INPUT_DIR / f"Noise_{i}.csv"
    if not fpath.exists():
        print(f"Warning: {fpath} not found.")
        continue
        
    with open(fpath, 'r', encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if i == 0:
            for row in reader:
                freq.append(float(row['Hz'])/10**3)
                tmp.append(float(row['dBc/Hz']))
            amp = list(np.zeros(len(tmp)))
        else:
            for row in reader:
                tmp.append(float(row['dBc/Hz']))
                
    total = [(amp[j] + tmp[j])/2 for j in range(len(tmp))]            
    amp = tmp
    tmp = []
                
plt.figure(figsize=(12,6))
plt.semilogx(freq, total, color='blue')
# vz_start = 5
# vz_end = 7.5
# vx_start = 9.5e+3
# vx_end = 11e+3
vz_cen = 7
vx_cen = 1.1e+3
video_bw= 10
plt.axvspan(vz_cen-video_bw/2, vz_cen+video_bw/2, facecolor='green', alpha=0.4)
plt.axvspan(vx_cen-video_bw/2, vx_cen+video_bw/2, facecolor='green', alpha=0.4)
peaks = [[0,0.5],[3,4],[5,6],[6.5,7.5],[8,9],[11,12],[19.2,19.8], [20.1, 20.6], [21.2,21.9],[22,23]]
for xx in peaks:
    plt.axvspan(xx[0], xx[1], facecolor='yellow', alpha=0.6)
plt.xlabel('Frequency [kHz]')
plt.ylabel('Phase noise [dBc/Hz]')
plt.xlim(10**-2,10**4)
save_path = OUTPUT_DIR / 'test.png'
plt.savefig(save_path, dpi=1000)
print(f"Plot saved to: {save_path}")
plt.show()
