import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Output periodicity in time steps
IT_DISPLAY = 10

# MODEL
# Model dimensions, [m]
nx = 401
nz = 401
dx = 10
dz = 10

# Elastic parameters
vp = 3300.0 * np.ones([nz, nx])         # velocity of compressional waves, [m/s]
vs = vp / 1.732                    # velocity of shear waves, [m/s]
rho = 2800.0 * np.ones([nz,nx])      # density, [kg/m3]

# Lame parameters
lam = rho*(vp**2 - 2*vs**2)       # first Lame parameter
mu = rho*vs**2                    # shear modulus, [N/m2]

## TIME STEPPING
t_total = .6                       # [sec] recording duration
dt = 0.8/(np.max(vp) * np.sqrt(1.0/dx**2 + 1.0/dz**2))
nt = round(t_total/dt)             # number of time steps
t = np.arange(0,nt)*dt
CFL = np.max(vp)*dt * np.sqrt(1.0/dx**2 + 1.0/dz**2)
## SOURCE
f0 = 10.0                          # dominant frequency of the wavelet
t0 = 1.20 / f0                     # excitation time
factor = 1e10                      # amplitude coefficient
angle_force = 90.0                 # spatial orientation

jsrc = round(nz/2)                 # source location along OZ
isrc = round(nx/2)                 # source location along OX

a = pi*pi*f0*f0
dt2rho_src = dt**2/rho[jsrc, isrc]  
source_term = factor * np.exp(-a*(t-t0)**2)                             # Gaussian
# source_term =  -factor*2.0*a*(t-t0)*exp(-a*(t-t0)**2)                # First derivative of a Gaussian:
# source_term = factor * (1.0 - 2.0*a*(t-t0).**2).*exp(-a*(t-t0).**2)        # Ricker source time function (second derivative of a Gaussian):

force_x = np.sin(angle_force * pi / 180) * source_term * dt2rho_src / (dx * dz)
force_z = np.cos(angle_force * pi / 180) * source_term * dt2rho_src / (dx * dz)
min_wavelengh = 0.5*np.min(vs)/f0     # shortest wavelength bounded by velocity in the air

## ABSORBING BOUNDARY (ABS)
abs_thick = min(np.floor(0.15*nx), np.floor(0.15*nz))         # thicknes of the layer
abs_rate = 0.3/abs_thick      # decay rate

lmargin = [abs_thick,abs_thick]
rmargin = [abs_thick,abs_thick]
weights = np.ones([nz+2,nx+2])
for iz in range(0,nz+2):
    for ix in range(0,nx+2):
        i = 0
        k = 0
        if (ix < lmargin[0] + 1):
            i = lmargin[0] + 1 - ix
        
        if (iz < lmargin[1] + 1):
            k = lmargin[1] + 1 - iz
        
        if (nx - rmargin[0] < ix):
            i = ix - nx + rmargin[0]
        
        if (nz - rmargin[1] < iz):
            k = iz - nz + rmargin[1]
        
        if (i == 0 and k == 0):
            continue
        
        rr = abs_rate * abs_rate * (i*i + k*k)
        weights[iz,ix] = np.exp(-rr)
    


## SUMMARY

## ALLOCATE MEMORY FOR WAVEFIELD
ux3 = np.zeros([nz+2,nx+2])            # Wavefields at t
uz3 = np.zeros([nz+2,nx+2])
ux2 = np.zeros([nz+2,nx+2])            # Wavefields at t-1
uz2 = np.zeros([nz+2,nx+2])
ux1 = np.zeros([nz+2,nx+2])            # Wavefields at t-2
uz1 = np.zeros([nz+2,nx+2])
# Coefficients for derivatives
co_dxx = 1/dx**2
co_dzz = 1/dz**2
co_dxz = 1/(4.0 * dx * dz)
co_dzx = 1/(4.0 * dx * dz)
dt2rho=(dt**2)/rho
lam_2mu = lam + 2 * mu

## Loop over TIME   
ims = []
fig, ax = plt.subplots()

for it in range(0, nt):
    ux3 = np.zeros(ux2.shape)
    uz3 = np.zeros(uz2.shape)
    # Second-order derivatives
    # Ux
    dux_dxx = co_dxx * (ux2[1:-1,0:-2] - 2*ux2[1:-1,1:-1] + ux2[1:-1,2:])
    dux_dzz = co_dzz * (ux2[0:-2,1:-1] - 2*ux2[1:-1,1:-1] + ux2[2:,1:-1])
    dux_dxz = co_dxz * (ux2[0:-2,2:] - ux2[2:,2:]- ux2[0:-2,0:-2] + ux2[2:,0:-2])
    # Uz

    duz_dxx = co_dxx * (uz2[1:-1,0:-2] - 2*uz2[1:-1,1:-1] + uz2[1:-1,2:])
    duz_dzz = co_dzz * (uz2[0:-2,1:-1] - 2*uz2[1:-1,1:-1] + uz2[2:,1:-1])
    duz_dxz = co_dxz * (uz2[0:-2,2:] - uz2[2:,2:]- uz2[0:-2,0:-2] + uz2[2:,0:-2])

    # Stress G
    sigmas_ux = lam_2mu * dux_dxx + lam * duz_dxz + mu * (dux_dzz + duz_dxz)
    sigmas_uz = mu * (dux_dxz + duz_dxx) + lam * dux_dxz + lam_2mu * duz_dzz
    # U(t) = 2*U(t-1) - U(t-2) + G dt2/rho
    ux3[1:-1,1:-1] = 2.0*ux2[1:-1,1:-1] - ux1[1:-1,1:-1] + sigmas_ux*dt2rho
    uz3[1:-1,1:-1] = 2.0*uz2[1:-1,1:-1] - uz1[1:-1,1:-1] + sigmas_uz*dt2rho
    # Add source tera
    ux3[jsrc, isrc] = ux3[jsrc, isrc] + force_x[it]
    uz3[jsrc, isrc] = uz3[jsrc, isrc] + force_z[it]
    # Exchange data between t-2 (1), t-1 (2) and t (3) and apply ABS
    ux1 = ux2 * weights
    ux2 = ux3 * weights
    uz1 = uz2 * weights
    uz2 = uz3 * weights
    # Output
    if it%IT_DISPLAY == 0:
        print('Time step: {0}, time: {1} '.format(it, t[it]))
        u=np.sqrt(ux3**2 + uz3**2)
        im = ax.imshow(u, animated=True)
        ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)


plt.show()