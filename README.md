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
## Example Usage:
`python3 main.py --upscale_factor 4 --batchSize 4 --testBatchSize 3 --nEpochs 30 --lr 0.001 --data <path_to_images>`
