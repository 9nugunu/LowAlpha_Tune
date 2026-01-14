from turtle import end_fill
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 24
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.weight']='bold'
plt.rcParams['font.size'] = 20
# plt.rcParams['image.cmap'] = 'gist_heat' 
plt.rcParams['image.cmap'] = 'jet' 

# alpha = 
e = 1.602176634e-19
c = 299792458           # m/s
E_0 = 629e+6            # eV
gam = E_0/(0.511e+6)+1
T_0 = 48/c              # s
U_0 = 9.1e+3            # eV
V_rf = 0.5e+6           # V
f_rf = 500e+6           # Hz
w_rf = 2*np.pi*f_rf     # Hz

delta_ini = np.linspace(0.1e-4, 0.2e-4, num=10)
alpha_ini = np.linspace(0.1e-4, 1e-4, num=10)
z_ini = np.linspace(0.1e-15, 20e-15, num=10)

phi_s = np.pi-np.sin(U_0/(e*V_rf))

############################## meshgrid #######################################
def f(alpha,d_delta):
    w_s = (np.sqrt(-V_rf*w_rf*alpha*np.cos(phi_s)))/np.sqrt(E_0*T_0)

    return (alpha-1/gam**2)*c/w_s*d_delta # m

X, Y = np.meshgrid(alpha_ini,delta_ini)
test = f(X,Y)

plt.figure()
plt.contourf(X/1e-4,Y/1e-4,test/1e-6,100, cmap='turbo') # offset unit [um]
cbar = plt.colorbar()
cbar.set_label("offset [um]")
plt.xlabel('momentum compaction factor x 10$^{-4}$')
plt.ylabel('relative energy spread x 10$^{-4}$')
plt.show()


###############################################################
# alpha = 10e-8

# def f(z,alpha):
#     w_s = (np.sqrt(-V_rf*w_rf*alpha*np.cos(phi_s)))/np.sqrt(E_0*T_0)
#     return z*w_s/((alpha-1/gam**2)*c)

# new_delta = f(z_ini,alpha)
# plt.figure()
# plt.plot(z_ini,new_delta) # offset unit [um]
# plt.xlabel('z x 10$^{-4}$')
# plt.ylabel('relative energy spread x 10$^{-4}$')
# plt.show()

############################## delta #######################################

# z_n = 10e-15
# alpha = [10e-4, 10e-5]
# for i in alpha:
#     w_s = (np.sqrt(-V_rf*w_rf*i*np.cos(phi_s)))/np.sqrt(E_0*T_0)
#     delta = z_n*w_s/((i-1/gam**2)*c)
#     print(delta)



