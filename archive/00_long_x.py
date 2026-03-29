from turtle import end_fill
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

plt.rcParams['axes.labelsize'] = 24
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.weight']='bold'
plt.rcParams['font.size'] = 20
# plt.rcParams['image.cmap'] = 'gist_heat' 
plt.rcParams['image.cmap'] = 'jet' 

e = 1.602176634e-19
c = 299792458               # m/s   
E_0 = 629e+6                # eV
U_0 = 9.1e+3                # eV
T_0 = 48/c                  # s
gam = E_0/(0.511e+6)+1
beta_x = 10
mu_x0 = 3.18
V_rf = 0.5e+6               # V
f_rf = 500e+6               # Hz
w_rf = 2*np.pi*f_rf         # Hz
e_x = 190e-9                # m * rad

k = 1
n = 1

delta_ini = np.linspace(0.1e-4, 0.2e-4, num=10) #1e-5
alpha_ini = np.linspace(0.1e-4, 1e-4, num=10) #1e-5

phi_s = np.pi-np.sin(U_0/(e*V_rf))

def f(alpha,delta):
    w_s = (np.sqrt(-V_rf*w_rf*alpha*np.cos(phi_s)))/np.sqrt(E_0*T_0)

    mu_s = w_s*T_0/(2*np.pi)
    return np.sqrt(e_x*beta_x)*sp.j1(4*delta/mu_s)

X, Y = np.meshgrid(alpha_ini,delta_ini)
test = f(X,Y)

plt.figure()
plt.contourf(X/1e-4,Y/1e-4,test/1e-6,100, cmap='turbo')
cbar = plt.colorbar()
cbar.set_label("offset [um]")
plt.xlabel('momentum compaction factor x 10$^{-4}$')
plt.ylabel('relative energy spread x 10$^{-4}$')
plt.show()


# Fixed-delta line plot: offset vs. alpha
delta_values = [5e-6, 1e-5, 2e-5]  # choose a few deltas to compare
alpha_line = np.linspace(alpha_ini.min(), alpha_ini.max(), 200)

plt.figure()
for d in delta_values:
    offset_line = f(alpha_line, d)
    plt.plot(alpha_line/1e-4, offset_line/1e-6, marker='o', markersize=3, label=rf'$\delta = {d:.0e}$')

# 2 um baseline
plt.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='2 $\mu$m baseline')

plt.xlabel('momentum compaction factor x 10$^{-4}$')
plt.ylabel('offset [um]')
plt.title('Offset vs. $\\alpha_c$ for selected $\\delta$')
plt.grid(True, linestyle=':')
plt.legend()
plt.show()


