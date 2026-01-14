import sdds
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, ifftshift, fftshift
import imageio, os
from pathlib import Path
from matplotlib.font_manager import FontProperties

plt.rcParams['axes.labelsize'] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.weight']='bold'
plt.rcParams['font.size'] = 12
# plt.rcParams['image.cmap'] = 'gist_heat' 
plt.rcParams['image.cmap'] = 'jet' 

# original + vx0 + vz0 + sidebands
c0 = 2.9979*10**8
T0 = 48/c0
#1.1144067 1.107315 vs_x_run1
#1.121504
#1.092343 1e-5
#1.119529 1e-4
vx = 1.1144067
# 0.007082 run1
# 0.002807 1e-5
# 0.006236 1e-4
vz = 0.007082
bw = 0.002
scale = 10**-6
index = 0


def col_page(filename, column):

    temp = sdds.SDDS(0)

    temp.load(filename)

    col_data = temp.columnData[temp.columnName.index(column)]

    col_data_temp = np.array(col_data)

    return col_data_temp

def figure_generation(fdir, file):
    x = col_page(fdir+file, "x").ravel()

    xp = col_page(fdir+file, "xp").ravel()

    t = col_page(fdir+file, "dt").ravel()

    p = col_page(fdir+file, "p").ravel()

    # p0 = 1.230922029484751e-3
    
#     plt.figure()
#     plt.scatter(x.ravel(),xp.ravel())
#     plt.show()

#     # fig_01 = plt.figure(1)
#     # fig_01.set_figwidth(5.5)
#     # fig_01.set_figheight(8)

    return x, xp, t, p

def make_animation(x_, y_):
    filenames = []
    x_1 = []
    y_1 = []
    ijk = 0
    ################### Animation #########################
    while ijk <= 360:
        x_1.append(x_[ijk]/scale)
        y_1.append(y_[ijk]/scale)
        # save frame
        if ijk % 20 == 0:
            filename = f'{ijk}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.xlabel(r'$x (\mu m)$')
            plt.close()
        
        elif ijk % 10 == 3 or ijk % 10 == 7:
            filenames.append(filename)
            plt.savefig(filename)
            plt.close()
            
        # if ijk == 340:
        #     break
        #############################
        plt.scatter(x_1, y_1, s=0.8, color = 'red')
        # plt.plot(x_1, y_1, 'r', linewidth=0.3)
        plt.xlabel(r'$x (\mu m)$')
        plt.ylabel(r'$p_x (\mu rad)$')
        plt.xlim([-140, 140])
        plt.ylim([-20, 20])
        # plt.pause(0.05)
        ijk += 1
    
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        jj = 0
        for filename in filenames:
            print(jj)
            image = imageio.imread(filename)
            writer.append_data(image)
            jj +=1
    
    for filename in set(filenames):
        os.remove(filename)

def filtering_norm(freq_x, data, cen_f, cutoff, ttt):
    lowband = cen_f-cutoff/2.0
    highband = cen_f+cutoff/2.0
    tmp = data
    print(cen_f)
    # plt.semilogy(freq_x, np.abs(data))
    # plt.show()

    for i in range(0, len(freq_x)):
        if ttt == 2:
            if lowband-vz < np.abs(freq_x[i]) < highband-vz or lowband+vz < np.abs(freq_x[i]) < highband+vz:
                pass
            else:
                tmp[i] = 0
        else:
            if lowband < np.abs(freq_x[i]) < highband:
                pass
            else:
                tmp[i] = 0
    
    tmp_result = ifftshift(ifft(tmp))
    print(tmp_result)
    
    if index == 0:
        return tmp
    elif index == 1:
        return tmp_result
    else:
        return

def position_retrive(interval,x_data,y_data,v_x,v_z,bw_):
    # make_animation(x_data,y_data)    
    ttt = 0
    freq = fftshift(np.fft.fftfreq(len(x_data),interval))*10**-6
    xdata_fft = fftshift(fft(x_data))

    xdata_1 = filtering_norm(freq,fftshift(fft(x_data)),v_x,bw_, ttt)
    ydata_1 = filtering_norm(freq,fftshift(fft(y_data)),v_x,bw_, ttt)
    ttt += 1
    xdata_2 = filtering_norm(freq,fftshift(fft(x_data)),v_z,bw_, ttt)
    ydata_2 = filtering_norm(freq,fftshift(fft(y_data)),v_z,bw_, ttt)
    ttt += 1
    xdata_3 = filtering_norm(freq,fftshift(fft(x_data)),v_x,bw_, ttt)
    ydata_3 = filtering_norm(freq,fftshift(fft(y_data)),v_x,bw_, ttt)
    # plt.figure()
    # plt.plot([i for i in range(len(xdata_1))],xdata_1)
    # plt.plot([i for i in range(len(ydata_1))],ydata_1)
    # plt.show()
    
    if index == 0:
        plt.semilogy(freq, np.abs(xdata_fft))
        plt.semilogy(freq, np.abs(xdata_1))
        plt.semilogy(freq, np.abs(xdata_2))
        plt.semilogy(freq, np.abs(xdata_3))
        plt.xlabel('frequency f [MHz]')
        plt.ylabel('Amplitude')
        plt.plot(freq, np.abs(t_fft))
        plt.show()
    
    elif index == 1:
        # make_animation(xdata_1,ydata_1)
        # fig = plt.figure()
        # ax1 =fig.add_subplot(141)
        # ax1.scatter(x_data/scale, y_data/scale,s=0.1,color='red')
        # ax2 =fig.add_subplot(142,sharey=ax1)
        # ax2.scatter(xdata_1/scale,ydata_1/scale,s=0.01,color='green')
        # ax3 =fig.add_subplot(143)
        # ax3.scatter(xdata_2/scale,ydata_2/scale,s=0.01,color='blue')
        # ax4 =fig.add_subplot(144,sharey=ax3)
        # ax4.scatter(xdata_3/scale,ydata_3/scale,s=0.01,color='black')
        # ax2.set_xlabel(r'$x (\mu m)$', fontsize='large', fontweight='bold')
        # ax1.set_ylabel(r'$p_x (\mu rad)$', fontsize='large')

        # plt.subplots_adjust(vspace=.0)
        
        fig, ax = plt.subplots(1,4,figsize=(17,6))
        ax[0].scatter(x_data/scale, y_data/scale,s=0.1,color='red')
        ax[1].scatter(xdata_1/scale,ydata_1/scale,s=0.01,color='green')
        ax[2].scatter(xdata_2/scale,ydata_2/scale,s=0.01,color='blue')
        ax[3].scatter(xdata_3/scale,ydata_3/scale,s=0.01,color='black')
        ax[1].set_xlabel(r'$x (\mu m)$')
        ax[0].set_ylabel(r'$p_x (\mu rad)$')

        plt.tight_layout()
        output_dir = Path("output") / "MLS"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'test.png', dpi=1000)
        plt.show()

x, xp, t, p = figure_generation('./input/MLS_tune_elegant/','run1.w2')

freq = fftshift(np.fft.fftfreq(len(t),T0))*10**-6
t_fft = fftshift(fft(t))
t_data1 = fftshift(fft(t))
p_fft = fftshift(fft(p))
plt.figure()
plt.scatter(np.abs(t), np.abs(p),s=0.1,color='red')
plt.close()
position_retrive(T0,x,xp,vx,vz,bw)





# plt.subplot(2, 1, 1)
# plt.plot(x, xp)
# plt.subplot(2, 1, 2)
# plt.plot(t, p)
# plt.show()
