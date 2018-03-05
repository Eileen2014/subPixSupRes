# Superresolution using sub-pixel convolutional neural network
["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158)

```
usage: main.py [-h] [--upscale_factor UPSCALE_FACTOR] [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--seed SEED] [--data DATA]

Keras Super-Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
  --batchSize BATCHSIZE
                        training batch size
  --testBatchSize TESTBATCHSIZE
                        testing batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=0.01
  --seed SEED           random seed to use. Default=123
  --data DATA           Path to image data
```
This example trains a super-resolution network on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model_epoch_<epoch_number>.pth

## Example Usage:

### Train

`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`

### Super Resolve
`python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`
