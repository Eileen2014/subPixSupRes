import argparse
import model
import data

# Input image dimensions
img_rows, img_cols = 32, 32
img_dims = (img_rows, img_cols)
num_imgs = 7985

# Training settings
parser = argparse.ArgumentParser(description='Keras Super-Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=3, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data', type=str, default="/nobackup/imaging/FLGW_scans/subpixel/data/img_align_celeba",
                    help='Path to image data')
opt = parser.parse_args()

print(opt)

print('===> Building model')
tm = model.build_ESPCN(img_dims, opt.upscale_factor, lr=opt.lr)
tm.summary()

data_path = opt.data
tm.fit_generator(
    zip(data.get_input_set(img_dims, batch_size=opt.batchSize, seed=opt.seed, path=data_path),
        data.get_output_set(img_dims, opt.upscale_factor, batch_size=opt.batchSize, seed=opt.seed, path=data_path)),
    steps_per_epoch=num_imgs/10,
    epochs=opt.nEpochs,
    # validation_data=validation_generator,
    # validation_steps=800,
    use_multiprocessing=False)

# TODO replace that by a proper validation
# import cv2
import numpy as np
import matplotlib

#matplotlib.use('Qt5Agg', warn=False, force=True)
from matplotlib import pyplot as plt

# Configure batch size and retrieve one batch of images
for X_batch, Y_batch in zip(data.get_input_set(img_dims, batch_size=opt.testBatchSize, seed=444, path=data_path),
                            data.get_output_set(img_dims, opt.upscale_factor, batch_size=opt.batchSize, seed=444, path=data_path)):
    Yhat_batch = tm.predict(X_batch)
    # Show images
    for i in range(3):
        plt.subplot(3, 4, 1 + i * 4)
        plt.imshow(np.clip(X_batch[i], 0., 1.))
        # plt.subplot(3, 4, 2 + i * 4)
        # res = cv2.resize(X_batch[i], None, fx=opt.upscale_factor, fy=opt.upscale_factor, interpolation=cv2.INTER_CUBIC)
        # plt.imshow(np.clip(res, 0., 1.))
        plt.subplot(3, 4, 3 + i * 4)
        plt.imshow(np.clip(Yhat_batch[i], 0., 1.))
        plt.subplot(3, 4, 4 + i * 4)
        plt.imshow(np.clip(Y_batch[i], 0., 1.))

    # show the plot
    plt.show()
    break
