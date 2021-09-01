import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from numba import float64

class Material:
    """
    Elastic wave properties for a material used in FDTD. All units are SI standard.
    VP - Primary wave speed, m/s
    VS - shear wave speed, m/s
    Rho - density, kg/m3
    Lam/mu - lame parameters
    """
    def __init__(self, vp=3300, vs=1905, rho = 2800):
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.lam = rho*(vp**2 - 2*vs**2)
        self.mu = rho*vs**2   
    
materials = {"steel":Material(vp=5960, vs=3235, rho=8000),
             "generic":Material(),
             "lightweightGeneric":Material(vp=596, vs=323, rho=800)}


@jit(nopython=True, parallel = True)
def stepAllCalcs(dx, dz, ux3, uz3, ux2, uz2, ux1, uz1, lam, mu, lam_2mu, dt2rho, weights):
    co_dxx = 1/dx**2
    co_dzz = 1/dz**2
    co_dxz = 1/(4.0 * dx * dz)

    #Ux
    dux_dxx = co_dxx * (ux2[1:-1,0:-2] - 2*ux2[1:-1,1:-1] + ux2[1:-1,2:])
    dux_dzz = co_dzz * (ux2[0:-2,1:-1] - 2*ux2[1:-1,1:-1] + ux2[2:,1:-1])
    dux_dxz = co_dxz * (ux2[0:-2,2:] - ux2[2:,2:]- ux2[0:-2,0:-2] + ux2[2:,0:-2])

    #Uz
    duz_dxx = co_dxx * (uz2[1:-1,0:-2] - 2*uz2[1:-1,1:-1] + uz2[1:-1,2:])
    duz_dzz = co_dzz * (uz2[0:-2,1:-1] - 2*uz2[1:-1,1:-1] + uz2[2:,1:-1])
    duz_dxz = co_dxz * (uz2[0:-2,2:] - uz2[2:,2:]- uz2[0:-2,0:-2] + uz2[2:,0:-2])

    # Stress G
    stressUX = lam_2mu * dux_dxx + lam * duz_dxz + mu * (dux_dzz + duz_dxz)
    stressUZ = mu * (dux_dxz + duz_dxx) + lam * dux_dxz + lam_2mu * duz_dzz
    # U(t) = 2*U(t-1) - U(t-2) + G dt2/rho
    ux3[1:-1,1:-1] = 2.0*ux2[1:-1,1:-1] - ux1[1:-1,1:-1] + stressUX*dt2rho
    uz3[1:-1,1:-1] = 2.0*uz2[1:-1,1:-1] - uz1[1:-1,1:-1] + stressUZ*dt2rho

    ux1 = ux2 * weights
    ux2 = ux3 * weights
    uz1 = uz2 * weights
    uz2 = uz3 * weights

    return [ux3, uz3, ux2, uz2, ux1, uz1]      

    

class FDTDModel:
    """
    Class representing an elastic wave simulation scenario
    """
    def __init__(self, sources,materialGrid=None, nx=400, nz=400, dx=10, dz=10, ntDisplay = 10):
        """
        tMax - max recording time
        materialGrid - NDArray of materials.
        sources - list of signal sources, formatted as [xPos, yPos, f(t)]
        """
        self.t=0
        self.nt = 0
        self.ntDisplay = ntDisplay
        #Number of nodes
        if materialGrid is None:
            print("Filling grid")
            materialGrid = np.full([nx,nz], materials["steel"])
        self.nx = materialGrid.shape[0]
        self.nz = materialGrid.shape[1]
        
        #Material properties for each node
        self.rho = np.zeros([self.nx, self.nz])
        self.vp = np.zeros([self.nx, self.nz])
        self.vs = np.zeros([self.nx, self.nz])
        self.lam = np.zeros([self.nx, self.nz])
        self.mu = np.zeros([self.nx, self.nz])


        for iz in range(0,self.nz):
            for ix in range(0,self.nx):
                self.rho[ix,iz] = materialGrid[ix,iz].rho
                self.vp[ix,iz] = materialGrid[ix,iz].vp
                self.vs[ix,iz] = materialGrid[ix,iz].vs
                self.lam[ix,iz] = materialGrid[ix,iz].lam
                self.mu[ix,iz] = materialGrid[ix,iz].mu


        #Size of field
        self.dx = dx
        self.dz = dz
        #Max primary wave speed dictates DT
        self.maxVP = np.max(self.vp)
        self.dt = 0.8/( self.maxVP * np.sqrt(1.0/self.dx**2 + 1.0/self.dz**2))
        self.CFL =  self.maxVP*self.dt * np.sqrt(1.0/self.dx**2 + 1.0/self.dz**2)
        self.sources=sources

        #Boundary - no reflections 
        self.abs_thick = min(np.floor(0.15*self.nx), np.floor(0.15*self.nz))
        self.abs_rate = 0.3/self.abs_thick

        #Field setup 
        self.weights = np.ones([self.nz+2,self.nx+2])
        for iz in range(0,self.nz+2):
            for ix in range(0,self.nx+2):
                i = 0
                k = 0
                if (ix < self.abs_thick + 1):
                    i = self.abs_thick + 1 - ix
                
                if (iz < self.abs_thick + 1):
                    k = self.abs_thick + 1 - iz
                
                if (self.nx - self.abs_thick < ix):
                    i = ix - self.nx + self.abs_thick
                
                if (self.nz - self.abs_thick < iz):
                    k = iz - self.nz + self.abs_thick
                
                if (i == 0 and k == 0):
                    continue
                
                rr = self.abs_rate * self.abs_rate * (i*i + k*k)
                self.weights[iz,ix] = np.exp(-rr)
        #Array allocation
        ## ALLOCATE MEMORY FOR WAVEFIELD
        self.ux3 = np.zeros([self.nz+2,self.nx+2])            # Wavefields at t
        self.uz3 = np.zeros([self.nz+2,self.nx+2])
        self.ux2 = np.zeros([self.nz+2,self.nx+2])            # Wavefields at t-1
        self.uz2 = np.zeros([self.nz+2,self.nx+2])
        self.ux1 = np.zeros([self.nz+2,self.nx+2])            # Wavefields at t-2
        self.uz1 = np.zeros([self.nz+2,self.nx+2])
        # Coefficients for derivatives
        self.dt2rho=(self.dt**2)/self.rho
        self.lam_2mu = self.lam + 2 * self.mu
        #Visualization
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ims = []
        self.frameTime = []

        print("Initialized elastic FDTD field of size {0}".format(str(self.nx)+","+str(self.nz)))

    def timeStep(self):
        self.t += self.dt
        self.nt += 1
        realTime = time.time()

        #Numba jit calculations
        newVels = stepAllCalcs(self.dx, self.dz, self.ux3, self.uz3, self.ux2, self.uz2, 
                                self.ux1, self.uz1, self.lam, self.mu, self.lam_2mu, 
                                self.dt2rho, self.weights)
        
        self.ux3, self.uz3, self.ux2, self.uz2, self.ux1, self.uz1 = newVels

        #Source velocity
        for source in self.sources:
            v = source[2](self.t)
            self.ux2[source[0], source[1]] += v[0]
            self.uz2[source[0], source[1]] += v[1]


        if self.nt%self.ntDisplay == 0:
            print('Time step: {0}, time: {1} '.format(self.nt, self.t))
            u=np.sqrt(self.ux3**2 + self.uz3**2)
            im = self.ax.imshow(u, animated=True)
            self.ims.append([im])

        self.frameTime.append((time.time()-realTime)*1000)