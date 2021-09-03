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