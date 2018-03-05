from __future__ import print_function
import matplotlib

matplotlib.use('Qt5Agg', warn=False, force=True)
from matplotlib import pyplot as plt
import data

# Input image dimensions
img_rows, img_cols = 64, 64
img_dims = (img_rows, img_cols)
upscale_factor = 4

# Configure batch size and retrieve one batch of images
for X_batch, Y_batch in zip(data.get_input_set(img_dims),
                            data.get_output_set(img_dims, upscale_factor)):
    # Show 9 images
    for i in range(0, 4):
        plt.subplot(240 + 1 + i * 2)
        plt.imshow(X_batch[i])

        plt.subplot(240 + 2 + i * 2)
        plt.imshow(Y_batch[i])

    # show the plot
    plt.show()
    break
