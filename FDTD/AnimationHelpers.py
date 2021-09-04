def matplotlibAnimation(frames, nFrames=5):
    """
    Frames as list of NDArrays
    nFrames indicates the number of frames to skip / downscale - if nFrames = 5, only every 5th frame will be rendered
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    framesDownscaled = frames[::5]
    ims=[]
    fig, ax = plt.subplots()
    for i in range(len(framesDownscaled)):
        im = ax.imshow(framesDownscaled[i], animated=True)
        ims.append([im])

    return animation.ArtistAnimation(fig, ims, interval=50, blit=True)