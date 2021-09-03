from FDTD.FDTDElastic import FDTDElasticModel
from FDTD.Materials import materials
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import matplotlib.animation as animation

f0=10
t0 = 1.20 / f0  
steel = materials["generic"]

def sourceFunc(t):
    v = np.exp(-(((2*(t-2*t0)/(t0))**2)))*np.sin(2*pi*f0*t)*0.01
    return [0,v]

sources =[]
sources.append([300,300, sourceFunc])

materialGrid = np.full([400,400], materials["steel"])
materialGrid[200:250, 0::] = materials["generic"]
materialGrid[0::,200:250] = materials["lightweightGeneric"]

model = FDTDElasticModel(sources, materialGrid=materialGrid, dx=10, dz=10)

for i in range(1000):
    model.timeStep()
    
print("Average tick time: " + str(np.mean(model.frameTime[1::])))

ani = animation.ArtistAnimation(model.fig, model.ims, interval=50, blit=True,repeat_delay=1000)
ani.save('A:/im.gif')

plt.show()