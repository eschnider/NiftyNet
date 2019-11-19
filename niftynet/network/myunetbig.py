# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.linear_resize import LinearResizeLayer
from niftynet.utilities.util_common import look_up_operations


class MyUNet3DBig(TrainableLayer):
    """
    Implementation of No New-Net
      Isensee et al., "No New-Net", MICCAI BrainLesion Workshop 2018.

      The major changes between this and our standard 3d U-Net:
      * input size == output size: padded convs are used
      * leaky relu as non-linearity
      * reduced number of filters before upsampling
      * instance normalization (not batch)
      * fits 128x128x128 with batch size of 2 on one TitanX GPU for
      training
      * no learned upsampling: linear resizing. 
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='leakyrelu',
                 name='NoNewNet'):
        super(MyUNet3DBig, self).__init__(name=name)

        self.n_features = [30, 64, 128, 128]
        self.n_features_down = [8,16,32, 64]
        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))

    def layer_op(self, thru_tensor, is_training=True, **unused_kwargs):
        """

        :param thru_tensor: the input is modified in-place as it goes through the network
        :param is_training:
        :param unused_kwargs:
        :return:
        """
        # image_size  should be divisible by 16 because of max-pooling 4 times, 2x2x2
        assert layer_util.check_spatial_dims(thru_tensor, lambda x: x % 16 == 0)
        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features_down[0], self.n_features_down[0]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L1')
        thru_tensor, conv_1 = block_layer(thru_tensor, is_training)
        print(block_layer)

        # crop 256x256x256 from original 512x512x512 to start
        crop_op = CropLayer(border=128, name='cropping_input')
        conv_1 = crop_op(conv_1)

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features_down[1], self.n_features_down[1]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L2')
        thru_tensor, conv_2 = block_layer(thru_tensor, is_training)
        print(block_layer)

        # crop 128x128x128 from 256x256x256
        crop_op = CropLayer(border=64, name='cropping_input')
        conv_2 = crop_op(conv_2)


        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[2], self.n_features[2]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L3')
        thru_tensor, conv_3 = block_layer(thru_tensor, is_training)
        print(block_layer)

        # crop 64x64x64 from 128x128x128
        crop_op = CropLayer(border=32, name='cropping_input')
        conv_3 = crop_op(conv_3)

        block_layer = PeelBlock(tuple(self.n_features_down[2] for i in range(16)),
                                tuple(3 for i in range(16)),
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L4')
        thru_tensor = block_layer(thru_tensor, is_training)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[2], self.n_features[2]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='bottom')
        thru_tensor, _ = block_layer(thru_tensor, is_training)
        print(block_layer)


        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[2], self.n_features[1]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='R3')
        concat_3 = ElementwiseLayer('CONCAT')(conv_3, thru_tensor)
        thru_tensor, _ = block_layer(concat_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[1], self.n_features[0]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='R2')
        concat_2 = ElementwiseLayer('CONCAT')(conv_2, thru_tensor)
        thru_tensor, _ = block_layer(concat_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('NONE',
                                (self.n_features[0], self.n_features[0], self.num_classes),
                                (3, 3, 1),
                                with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='R1_FC')

        concat_1 = ElementwiseLayer('CONCAT')(conv_1, thru_tensor)
        thru_tensor, _ = block_layer(concat_1, is_training)
        print(block_layer)

        return thru_tensor


SUPPORTED_OP = {'DOWNSAMPLE', 'UPSAMPLE', 'NONE'}


class UNetBlock(TrainableLayer):
    def __init__(self,
                 func,
                 n_chns,
                 kernels,
                 w_initializer=None,
                 w_regularizer=None,
                 with_downsample_branch=False,
                 acti_func='leakyrelu',
                 name='UNet_block'):

        super(UNetBlock, self).__init__(name=name)

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)

        self.kernels = kernels
        self.n_chns = n_chns
        self.with_downsample_branch = with_downsample_branch
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, thru_tensor, is_training):
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            # no activation before final 1x1x1 conv layer 
            acti_func = self.acti_func if kernel_size > 1 else None
            feature_normalization = 'instance' if acti_func is not None else None

            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=acti_func,
                                         name='{}'.format(n_features),
                                         feature_normalization=feature_normalization)
            thru_tensor = conv_op(thru_tensor, is_training)

        if self.with_downsample_branch:
            branch_output = thru_tensor
        else:
            branch_output = None

        if self.func == 'DOWNSAMPLE':
            downsample_op = DownSampleLayer('MAX', kernel_size=2, stride=2, name='down_2x2')
            thru_tensor = downsample_op(thru_tensor)
        elif self.func == 'UPSAMPLE':
            up_shape = [2 * int(thru_tensor.shape[i]) for i in (1, 2, 3)]
            upsample_op = LinearResizeLayer(up_shape)
            thru_tensor = upsample_op(thru_tensor)

        elif self.func == 'NONE':
            pass  # do nothing
        return thru_tensor, branch_output

class PeelBlock(TrainableLayer):
    def __init__(self,
                 n_chns,
                 kernels,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='leakyrelu',
                 name='zoom_block'):
        super(PeelBlock, self).__init__(name=name)

        self.kernels = kernels
        self.n_chns = n_chns
        self.acti_func = acti_func
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, thru_tensor, is_training):
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            # no activation before final 1x1x1 conv layer
            acti_func = self.acti_func if kernel_size > 1 else None
            feature_normalization = 'instance' if acti_func is not None else None
            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         padding='VALID',
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=acti_func,
                                         name='{}'.format(n_features),
                                         feature_normalization=feature_normalization)
            thru_tensor = conv_op(thru_tensor, is_training)

        return thru_tensor
