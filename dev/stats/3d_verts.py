import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm

def main():

    # Parse
    parser = ArgumentParser()
    parser.add_argument('infile')
    args = parser.parse_args()

    # Load infile
    verts = np.loadtxt(args.infile)[:,:3]

    # Plot
    ax=Axes3D( plt.figure() )
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    ax.set_ylim(-100,100)
    ax.set_zlim(-100,100)

    # Setting color bar
    #color_map = cm.ScalarMappable(cmap=cm.Reds)
    #color_map.set_array( [] )
    #plt.colorbar(color_map)

    # Creating the heatmap
    ax.scatter(
                verts[:,2],
                verts[:,0],
                verts[:,1],
                marker='o',
                s=10,
                color='Red'
                )

    plt.show()

if __name__ == '__main__': main()
