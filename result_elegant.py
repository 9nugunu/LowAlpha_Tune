'''
Created on Jul 19, 2022

@author: mcf
'''
from _ast import If
'''
Created on Jul 9, 2022

@author: mcf
'''

import sdds, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, fftfreq, ifft, ifftshift
from scipy.signal import find_peaks
from pathlib import Path



plt.rcParams['axes.labelsize'] = 24
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.weight']='bold'
plt.rcParams['font.size'] = 20
# plt.rcParams['image.cmap'] = 'gist_heat' 
plt.rcParams['image.cmap'] = 'jet' 

c0 = 2.9979*10**8
T0 = 48/c0
p0 = 1.230922029484751*10**3
OUT_DIR = Path("output") / Path(__file__).stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

def col_page(filename, column):
    temp = sdds.SDDS(0)
    temp.load(filename)
    col_data = temp.columnData[temp.columnName.index(column)]
    col_data_temp = np.array(col_data)
    return col_data_temp

def amp_cal(p):
    fdir, valD, valA = p
    os.chdir(fdir)
    file =  'opt_A%.2e_D%.2e_check.w2'%(valA,valD)
    if os.path.isfile(file) :    
        x = col_page(file, "x")
        # xp = col_page(file, "xp")
        t = col_page(file, "dt")*c0
        # p = col_page(file, "p")
        freq = fftshift(fftfreq(len(x),T0))/10**6
        fz = fftshift(fft(t.ravel()))
        fx = fftshift(fft(x.ravel()))

        import matplotlib.pyplot as plt
        plt.plot(freq, np.abs(fz), c='b')
        plt.plot(freq, np.abs(fx), c='r')
        plt.show()
        # peaks, _ = find_peaks(np.abs(fx), prominence = 0.1, height = 0.2) #height=0.01)    
        peaks, _ = find_peaks(np.abs(fx), prominence = np.max(np.abs(fx))/20., height=np.max(np.abs(fx))/20., distance=3) #height=0.01)
        cal_peak = []
        for i in peaks:
            if freq[i] > 0 :                
                cal_peak.append([freq[i], np.abs(fx[i])])
            else:
                pass
        cal_peak = np.array(cal_peak)
        print(cal_peak)
        # cal_peak = cal_peak[cal_peak[:,0].argsort()]
        vxs,vzs = [],[]
        for k in range(len(cal_peak)):
            if 0.0 < cal_peak[k][0] < 0.1 :
                vzs.append(cal_peak[k])
            elif cal_peak[k][0] > 1 :
                vxs.append(cal_peak[k])
            else:
                pass
        vxs = np.array(vxs)
        vzs = np.array(vzs)
        print("vxs", vxs)
        print("vzs", vzs)
        if len(vxs) > 1:
            vxs = vxs[vxs[:,1].argsort()]
            vx = vxs[-1,0]
        else:
            vx = vxs[0,0]

        if len(vzs) > 1:
            vzs = vzs[vzs[:,1].argsort()]
            vz = vzs[-1,0]
        else:
            vz = vzs[0,0]
        print("*"*1000)
        print(vxs, vzs)
        # vxn1 = vx-vz
        # vxp1 = vx+vz
        # bw = vz/3.
        bw = 0.001 
        for i in range(len(freq)):
            if  vx-vz -bw/2 < np.abs(freq[i]) < vx-vz+bw/2 or vx+vz-bw/2 < np.abs(freq[i]) < vx+vz+bw/2 :
                fz[i] = 0
                # pass
            elif vz -bw/2 < np.abs(freq[i]) < vz + bw/2:
                fx[i] = 0
            else:
                fx[i] = 0
                fz[i] = 0
        
        filtered_x = np.real(ifftshift(ifft(fx)))
        filtered_z = np.real(ifftshift(ifft(fz)))
        print(fx)
        print(fz)
        print("*"*1000)
        return [valA/10**-4,valD/10**-4,np.max(np.abs(filtered_x))/10**-6, np.max(np.abs(filtered_z))/10**-6]
        # return [valA/10**-4,valD/10**-4,np.max(np.std(filtered_x))/10**-6, np.max(np.std(filtered_z))/10**-6]
    else:
        return [0,0,0,0]


if __name__ == '__main__':
    fdir = './input/results/scan_elegant/results/'
    scan_startD,scan_stopD,scan_stepD = 10**-5,2*10**-5,0.1*10**-5
    scan_startA,scan_stopA,scan_stepA = 10**-5,1*10**-4,1*10**-5

    # scan_startD,scan_stopD,scan_stepD = 10**-4, 2.2*10**-4, 0.2*10**-5
    # scan_startA,scan_stopA,scan_stepA = 10**-5, 1.01*10**-4, 0.1*10**-5

    alpha_new = np.arange(scan_startA,scan_stopA,scan_stepA)/10**-4
    delD_new = np.arange(scan_startD,scan_stopD,scan_stepD)/10**-4
    plotX, plotY = np.meshgrid(alpha_new, delD_new)

    if os.path.isfile(fdir+'X.txt') and os.path.isfile(fdir+'Z.txt'):
        # loading:
        X = np.loadtxt(fdir+'X.txt', unpack=True).T
        Z = np.loadtxt(fdir+'Z.txt', unpack=True).T
    
    else :
        new_value = []
        for i in np.arange(scan_startD,scan_stopD,scan_stepD):
            for k in np.arange(scan_startA,scan_stopA,scan_stepA):
                new_value.append([fdir, i, k])
        
        import multiprocessing
        processes = multiprocessing.cpu_count()
        
        if processes > 1:
            print ('multiprocessing works')
            mp_pool = multiprocessing.Pool(processes)
            results = np.array(mp_pool.map(amp_cal, new_value))
            mp_pool.close()
            mp_pool.join()
        else :
            results=np.array([amp_cal(new_value[i]) for i in range(len(new_value))])
    
        results = results.reshape((-1,4))
        # print (results)
        Z = np.zeros((len(delD_new),len(alpha_new)))
        X = np.zeros((len(delD_new),len(alpha_new)))
    
    
        for i in range(len(alpha_new)):
            for k in range(len(delD_new)):
                for l in range(len(results)):
                    if results[l][0] == plotX[k][i] and results[l][1] == plotY[k][i]:
                        X[k][i] = results[l][2]
                        Z[k][i] = results[l][3]
                    else:
                        pass
    
        np.savetxt(fdir+'X.txt', X)
        np.savetxt(fdir+'Z.txt', Z)
                
        
    # Plot Z contour
    fig,ax=plt.subplots(1,1,figsize=(12,9))
    cp = ax.contourf(plotX, plotY, Z, levels=100)
    cbar = fig.colorbar(cp) # Add a colorbar to a plot
    cbar.set_label(r'offset ($\mu$m)')
    plt.xlabel(r'momentum compaction factor $\times 10^{-4}$')
    plt.ylabel(r'relative energy spread $\times 10^{-4}$')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Z_offset_contour.png", dpi=300)
    
    # Plot X contour
    fig,ax=plt.subplots(1,1,figsize=(12,9))
    cp = ax.contourf(plotX, plotY, X, levels=100)
    cbar = fig.colorbar(cp) # Add a colorbar to a plot
    cbar.set_label(r'offset ($\mu$m)')
    plt.xlabel(r'momentum compaction factor $\times 10^{-4}$')
    plt.ylabel(r'relative energy spread $\times 10^{-4}$')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "X_offset_contour.png", dpi=300)

    # Side-by-side with unified color scale
    vmin = min(X.min(), Z.min())
    vmax = max(X.max(), Z.max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cf1 = axes[0].contourf(plotX, plotY, Z, levels=100, vmin=vmin, vmax=vmax)
    axes[0].set_title("Z offset")
    axes[0].set_xlabel(r'$\alpha_c \times 10^{-4}$')
    axes[0].set_ylabel(r'$\delta \times 10^{-4}$')
    cf2 = axes[1].contourf(plotX, plotY, X, levels=100, vmin=vmin, vmax=vmax)
    axes[1].set_title("X offset")
    axes[1].set_xlabel(r'$\alpha_c \times 10^{-4}$')
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cbar = fig.colorbar(cf1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label(r'offset ($\mu$m)')
    fig.savefig(OUT_DIR / "XZ_offset_contour_side_by_side.png", dpi=300)
    plt.show()
