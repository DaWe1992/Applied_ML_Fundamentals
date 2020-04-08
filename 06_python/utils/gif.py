# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:08:52 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import imageio
import matplotlib.pyplot as plt

from io import BytesIO


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def make_gif(figures, filename="test.gif", fps=10, **kwargs):
    """
    Generates a GIF file.
    
    :param figures:             list of figures
    :param filename:            name of the GIF file
    :param fps:
    """
    images = []
    
    for fig in figures:
        output = BytesIO()
        fig.savefig(output)
        plt.close(fig)  
        output.seek(0)
        images.append(imageio.imread(output))
        
    imageio.mimsave(filename, images, fps=fps, **kwargs)