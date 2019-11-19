# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.upsample import UpSampleLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.network.highres3dnet import HighResBlock


class Ownnet2b(BaseNet):
    """
    ### Description
    reimplementation of DeepMedic:
        Kamnitsas et al., "Efficient multi-scale 3D CNN with fully connected
        CRF for accurate brain lesion segmentation", MedIA '17

    ### Building blocks
    [CONV]          - 3x3x3 convolutional layer
    [denseCONV]     - 1x1x1 convolutional layer

    ### Diagram
    INPUT --> CROP -------> [CONV]x8 ------> [SUM] ----> [denseCONV]x3 --> OUTPUT
                |                             |
            DOWNSAMPLE ---> [CONV]x8 ---> UPSAMPLE


    ### Constraints:
    - The downsampling factor (d_factor) should be odd
    - Label size = [(image_size / d_factor) - 16]* d_factor
    - Image size should be divisible by d_factor

    # Examples:
    - Appropriate configuration for training:
    image spatial window size = 57, label spatial window size = 9, d_ factor = 3
    - Appropriate configuration for inference:
    image spatial window size = 105, label spatial window size = 57, d_ factor = 3
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name="Ownnet"):
        """

        :param num_classes: int, number of channels of output
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(Ownnet2b, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_down_0', 'n_features': 4, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 8, 'kernels': (3, 3), 'repeat': 1},
            {'name': 'res_2', 'n_features': 16, 'kernels': (3, 3), 'repeat': 1},
            {'name': 'res_3', 'n_features': 32, 'kernels': (3, 3), 'repeat': 1},
            {'name': 'conv_up_0', 'n_features': 8, 'kernel_size': 3},
            {'name': 'res_4', 'n_features': 8, 'kernels': (3, 3), 'repeat': 1},
            # {'name': 'res_4', 'n_features': 8, 'kernels': (3, 3)},
            {'name': 'conv_class', 'n_features': num_classes, 'kernel_size': 1}]

        self.wide_layers = [
            {'name': 'conv_down_1_0', 'n_features': 8, 'kernels': (3), 'repeat': 1},
            {'name': 'conv_1_0', 'n_features': 16, 'kernels': (3, 3, 3, 3), 'repeat': 1},
            {'name': 'conv_down_1_1', 'n_features': 32, 'kernels': (3), 'repeat': 1},
            {'name': 'conv_1_1', 'n_features': 32, 'kernels': (3, 3), 'repeat': 1},
            {'name': 'conv_up_1', 'n_features': 8, 'kernel_size': 3, 'repeat': 1}]

        self.crop_diff = 32


    def layer_op(self, images, is_training, layer_id=-1, **unused_kwargs):
        """

        :param images: tensor, input to the network, size should be divisible by d_factor
        :param is_training: boolean, True if network is in training mode
        :param layer_id: not in use
        :param unused_kwargs:
        :return: tensor, network output
        """
        # image_size is defined as the largest context, then:
        #   downsampled path size: image_size / d_factor
        #   downsampled path output: image_size / d_factor - 16

        # to make sure same size of feature maps from both pathways:
        #   normal path size: (image_size / d_factor - 16) * d_factor + 16
        #   normal path output: (image_size / d_factor - 16) * d_factor

        # where 16 is fixed by the receptive field of conv layers
        # TODO: make sure label_size = image_size/d_factor - 16

        # image_size has to be an odd number and divisible by 3 and
        # smaller than the smallest image size of the input volumes

        # label_size should be (image_size/d_factor - 16) * d_factor

        # assert self.d_factor % 2 == 1  # to make the downsampling centered
        # assert (layer_util.check_spatial_dims(
        #     images, lambda x: x % self.d_factor == 0))
        # assert (layer_util.check_spatial_dims(
        #     images, lambda x: x % 2 == 1))  # to make the crop centered
        # assert (layer_util.check_spatial_dims(
        #     images,
        #     lambda x: x > self.d_factor * 16))  # required by receptive field

        layer_instances = []

        # crop 128x128x128 from 256x256x256 to make a skip connection later
        crop_op = CropLayer(border=64, name='cropping_input')
        full_res_flow = crop_op(images)
        layer_instances.append((crop_op, full_res_flow))

        ### first common down convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=2,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        # crop 64x64x64 from 128x128x128
        crop_op = CropLayer(border=self.crop_diff, name='cropping_downsamplet_input')
        narrow_flow = crop_op(flow)
        layer_instances.append((crop_op, narrow_flow))

        ### wide path down convolution layer, (128 -> 64), dilated by 4
        params = self.wide_layers[0]
        with DilatedTensor(flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    with_res=False,
                    stride=2,
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        wide_flow = dilated.tensor

        ### wide path res block 1, dilated by 2
        # print(tf.size(wide_flow))
        params = self.wide_layers[1]
        with DilatedTensor(wide_flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    with_res=False,
                    stride=1,
                    padding='VALID',
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        wide_flow = dilated.tensor

        ### wide path convolution layer, only wide path (56 -> 28), dilated by 4
        params = self.wide_layers[2]
        with DilatedTensor(wide_flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    with_res=False,
                    stride=2,
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        wide_flow = dilated.tensor

        ### wide path res block 2, dilated by 3
        params = self.wide_layers[3]
        with DilatedTensor(wide_flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    with_res=False,
                    stride=1,
                    padding='VALID',
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        wide_flow = dilated.tensor

        ### wide path 3x3x3 deconvolution layer
        params = self.wide_layers[4]
        fc_layer = DeconvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=3,
            stride=4,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='deconv')
        wide_flow = fc_layer(wide_flow, is_training)
        layer_instances.append((fc_layer, wide_flow))



        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[1]
        with DilatedTensor(narrow_flow, dilation_factor=1) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        narrow_flow = dilated.tensor

        ### resblocks, all kernels dilated by 2
        params = self.layers[2]
        with DilatedTensor(narrow_flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        narrow_flow = dilated.tensor

        ### resblocks, all kernels dilated by 4
        params = self.layers[3]
        with DilatedTensor(narrow_flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        narrow_flow = dilated.tensor

        # concatenate both pathways
        common_flow = ElementwiseLayer('CONCAT')(narrow_flow, wide_flow)

        ### 3x3x3 deconvolution layer
        params = self.layers[4]
        fc_layer = DeconvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=3,
            stride=2,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='deconv')
        common_flow = fc_layer(common_flow, is_training)
        layer_instances.append((fc_layer, common_flow))

        # skip connection
        common_flow = ElementwiseLayer('CONCAT')(full_res_flow, common_flow)

        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[5]
        with DilatedTensor(common_flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        common_flow = dilated.tensor

        # ### 3x3x3 convolution layer
        # params = self.layers[5]
        # last_conv_layer = ConvolutionalLayer(
        #     n_output_chns=params['n_features'],
        #     kernel_size=params['kernel_size'],
        #     acti_func=self.acti_func,
        #     w_initializer=self.initializers['w'],
        #     w_regularizer=self.regularizers['w'],
        #     name=params['name'])
        # common_flow = last_conv_layer(common_flow, is_training)
        # layer_instances.append((last_conv_layer, common_flow))

        ### 1x1x1 convolution layer
        params = self.layers[6]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        common_flow = fc_layer(common_flow, is_training)
        layer_instances.append((fc_layer, common_flow))

        # set training properties
        if is_training:
            self._print(layer_instances)
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)


class HighResBlock(TrainableLayer):
    """
    This class defines a high-resolution block with residual connections
    kernels

        - specify kernel sizes of each convolutional layer
        - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5

    with_res

        - whether to add residual connections to bypass the conv layers
    """

    def __init__(self,
                 n_output_chns,
                 kernels=(3, 3),
                 acti_func='relu',
                 stride=1,
                 padding='SAME',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='HighResBlock'):
        """

        :param n_output_chns: int, number of output channels
        :param kernels: list of layer kernel sizes
        :param acti_func: activation function to use
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param with_res: boolean, set to True if residual connection are to use
        :param name: layer name
        """

        super(HighResBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_func = acti_func
        self.with_res = with_res
        self.stride = stride
        self.padding = padding

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        """

        :param input_tensor: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :return: tensor, output of the residual block
        """
        output_tensor = input_tensor
        for (i, k) in enumerate(self.kernels):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.regularizers['w'],
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                padding=self.padding,
                                kernel_size=k,
                                stride=self.stride,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            output_tensor = bn_op(output_tensor, is_training)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor
