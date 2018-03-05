from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.activations import tanh


def resize(image, img_size):
    # import cv2
    # image = cv2.resize(image, (66, 200))
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(image, img_size, method=ktf.image.ResizeMethod.BILINEAR)
    # import tensorflow as tf
    # import keras.backend as kb
    # resized = tf.image.resize_images(image, factor, factor, kb.image_data_format())
    return resized


def build_SRCNN(img_dims, upscale_factor, lr=0.001):
    rs_size = tuple(upscale_factor * x for x in img_dims)
    inputs = Input(shape=img_dims + (3,))
    rs = Lambda(resize,
                arguments={'img_size': rs_size},
                input_shape=img_dims + (3,), output_shape=rs_size + (3,)
                )(inputs)
    c1 = Conv2D(64, kernel_size=9, activation='relu', padding='same', name="c1")(rs)
    c2 = Conv2D(32, kernel_size=1, activation='relu', padding='same', name="c2")(c1)
    c3 = Conv2D(3, kernel_size=5, padding='same', name="c3")(c2)
    m = Model(input=inputs, output=c3)
    m.compile(Adam(lr=lr), 'mse')
    return m


def build_SRCNN2(img_dims, upscale_factor, lr=0.001):
    rs_size = tuple(upscale_factor * x for x in img_dims)
    inputs = Input(shape=img_dims + (3,))
    rs = Lambda(resize,
                arguments={'img_size': rs_size},
                input_shape=img_dims + (3,), output_shape=rs_size + (3,)
                )(inputs)
    c1 = Conv2D(64, kernel_size=5, strides=1, activation='relu', padding='same', name="c1")(rs)
    c2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same', name="c2")(c1)
    c3 = Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same', name="c3")(c2)
    c4 = Conv2D(3, kernel_size=3, strides=1, padding='same', name="c4")(c3)
    m = Model(input=inputs, output=c4)
    m.compile(Adam(lr=lr), 'mse')
    return m


def subpixel_shape(input_shape, scale):
    dims = [input_shape[1] * scale,
            input_shape[2] * scale,
            int(input_shape[3] / (scale ** 2))]
    output_shape = tuple(dims)
    return output_shape


def subpixel(x, scale):
    import tensorflow as tf
    return tf.depth_to_space(x, scale)


def build_SPSRCNN(img_dims, upscale_factor, lr=0.001):
    inputs = Input(shape=img_dims + (3,))
    c1 = Conv2D(64, kernel_size=5, strides=1, activation='relu', padding='same', name="c1")(inputs)
    c2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same', name="c2")(c1)
    c3 = Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same', name="c3")(c2)
    c4 = Conv2D(upscale_factor ** 2 * 3, kernel_size=3, strides=1, padding='same', name="c4")(c3)
    s = Lambda(subpixel,
               arguments={'scale': upscale_factor},
               output_shape=subpixel_shape(c4._keras_shape, upscale_factor),
               name='subpixel')(c4)
    m = Model(input=inputs, output=s)
    m.compile(Adam(lr=lr), 'mse')
    return m


def build_SPSRCNN2(img_dims, upscale_factor, lr=0.001):
    inputs = Input(shape=img_dims + (3,))
    c1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name="c1")(inputs)
    c2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name="c2")(c1)
    c3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name="c3")(c2)
    c4 = Conv2D(upscale_factor ** 2 * 3, kernel_size=3, padding='same', name="c4")(c3)
    s = Lambda(subpixel,
               arguments={'scale': upscale_factor},
               output_shape=subpixel_shape(c4._keras_shape, upscale_factor),
               name='subpixel')(c4)
    m = Model(input=inputs, output=s)
    m.compile(Adam(lr=lr), 'mse')
    return m


def build_ESPCN(img_dims, upscale_factor, lr=0.001):
    inputs = Input(shape=img_dims + (3,))
    c1 = Conv2DTranspose(64, kernel_size=1, activation='linear', padding='same', name="c1")(inputs)
    l1 = LeakyReLU(alpha=.2)(c1)
    c2 = Conv2DTranspose(64, kernel_size=5, activation='linear', padding='same', name="c2")(l1)
    l2 = LeakyReLU(alpha=.2)(c2)
    # c3 = Conv2DTranspose(64, kernel_size=3, activation='linear', padding='same', name="c3")(l2)
    # l3 = LeakyReLU(alpha=.001)(c3)
    c4 = Conv2DTranspose(upscale_factor ** 2 * 3, kernel_size=5, padding='same', name="c4")(l2)
    s = Lambda(subpixel,
               arguments={'scale': upscale_factor},
               output_shape=subpixel_shape(c4._keras_shape, upscale_factor),
               name='subpixel')(c4)
    t = Activation(tanh)(s)
    m = Model(input=inputs, output=t)
    m.compile(Adam(lr=lr), 'mse')
    return m


def _phase_shift(I, r, shape):
    import tensorflow as tf
    # bsize, a, b, c = I.get_shape().as_list()
    # bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    _, a, b, _ = shape
    bsize = 2  # TODO Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, shape, color=True):
    import tensorflow as tf
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r, shape) for x in Xc], axis=3)
    else:
        X = _phase_shift(X, r, shape)
    return X


def build_ESPCN2(img_dims, upscale_factor, lr=0.001):
    inputs = Input(shape=img_dims + (3,))
    c1 = Conv2DTranspose(64, kernel_size=1, activation='linear', padding='same', name="c1")(inputs)
    l1 = LeakyReLU(alpha=.2)(c1)
    c2 = Conv2DTranspose(64, kernel_size=5, activation='linear', padding='same', name="c2")(l1)
    l2 = LeakyReLU(alpha=.2)(c2)
    # c3 = Conv2DTranspose(64, kernel_size=3, activation='linear', padding='same', name="c3")(l2)
    # l3 = LeakyReLU(alpha=.001)(c3)
    c4 = Conv2DTranspose(upscale_factor ** 2 * 3, kernel_size=5, padding='same', name="c4")(l2)
    s = Lambda(PS,
               arguments={'r': upscale_factor, 'shape': c4._keras_shape},
               output_shape=subpixel_shape(c4._keras_shape, upscale_factor),
               name='subpixel')(c4)
    t = Activation(tanh)(s)
    m = Model(input=inputs, output=t)
    m.compile(Adam(lr=lr), 'mse')
    return m


'''
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)
'''
