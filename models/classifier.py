import torch
import torch.nn as nn
from .resnet_block import ResNetBlock
import models.torch_utils


class Classifier(torch.nn.Module):
    """
    Base classifier.
    """

    def __init__(self, N_class, resolution, **kwargs):
        """
        Initialize classifier.

        The keyword arguments, resolution, number of classes and other architecture parameters
        from subclasses are saved as attributes. This allows to easily save and load the model
        using common.state without knowing the exact architecture in advance.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution
        :type resolution: [int]
        """

        super(Classifier, self).__init__()

        assert N_class > 0, 'positive N_class expected'
        assert len(resolution) <= 3

        self.N_class = int(N_class)  # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        """ (int) Number of classes. """

        self.resolution = list(resolution)
        """ ([int]) Resolution as (channels, height, width) """

        self.kwargs = kwargs
        """ (dict) Kwargs. """

        self.include_clamp = self.kwargs_get('clamp', True)
        """ (bool) Whether to apply input clamping. """

        self.include_whiten = self.kwargs_get('whiten', False)
        """ (bool) Whether to apply input whitening/normalization. """

        self.include_scale = self.kwargs_get('scale', False)
        """ (bool) Whether to apply input scaling. """

        # __ attributes are private, which is important for the State to work properly.
        self.__layers = []
        """ ([str]) Will hold layer names. """

        self._N_output = self.N_class if self.N_class > 2 else 1
        """ (int) Number of outputs. """

        if self.include_clamp:
            self.append_layer('clamp', models.torch_utils.Clamp())

        assert not (self.include_whiten and self.include_scale)

        if self.include_whiten:
            # Note that the weight and bias needs to set manually corresponding to mean and std!
            whiten = models.torch_utils.Normalize(resolution[0])
            self.append_layer('whiten', whiten)

        if self.include_scale:
            # Note that the weight and bias needs to set manually!
            scale = models.torch_utils.Scale(1)
            scale.weight.data[0] = -1
            scale.bias.data[0] = 1
            self.append_layer('scale', scale)

    def kwargs_get(self, key, default):
        """
        Get argument if not None.

        :param key: key
        :type key: str
        :param default: default value
        :type default: mixed
        :return: value
        :rtype: mixed
        """

        value = self.kwargs.get(key, default)
        if value is None:
            value = default
        return value

    def append_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.append(name)

    def prepend_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        self.insert_layer(0, name, layer)

    def insert_layer(self, index, name, layer):
        """
        Add a layer.

        :param index: index
        :type index: int
        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.insert(index, name)

    def forward(self, image, features=False, return_features=False):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :param features: whether to return only the representation layer
        :param return_features: whether to also return representation layer
        :type return_features: bool
        :return: logits
        :rtype: torch.autograd.Variable
        """

        out_features = []
        output = image

        # separate loops for memory constraints
        if features:
            for name in self.__layers[:-1]:
                output = getattr(self, name)(output)
            return output
        elif return_features:
            for name in self.__layers:
                output = getattr(self, name)(output)
                out_features.append(output)
            return output, out_features
        else:
            for name in self.__layers:
                output = getattr(self, name)(output)
            return output

    def layers(self):
        """
        Get layer names.

        :return: layer names
        :rtype: [str]
        """

        return self.__layers

    def __str__(self):
        """
        Print network.
        """

        string = ''
        for name in self.__layers:
            string += '(' + name + ', ' + getattr(self, name).__class__.__name__ + ')\n'
            if isinstance(getattr(self, name), torch.nn.Sequential) or isinstance(getattr(self, name), ResNetBlock):
                for module in getattr(self, name).modules():
                    string += '\t(' + module.__class__.__name__ + ')\n'
        return string


def MLP(classes):
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, classes)
    )

class DenseNet(nn.Module):
    def __init__(self, classes=3, ndim=2, layers=5, out_size=2, **kwargs):
        super().__init__()

        blocks = []

        in_dim = ndim

        for i in range(layers):
            block = nn.Sequential(
                nn.Linear(in_dim, out_size),
                nn.ReLU(),
            )

            in_dim += out_size

            blocks.append(block) 

        blocks.append(nn.Linear(in_dim, classes))

        self.layers = nn.ModuleList(blocks)

    def __call__(self, batch, features=False):
        inputs = batch

        layers = self.layers if not features else self.layers[:-1]

        for layer in layers:
            outputs = layer(inputs)
            inputs = torch.cat((inputs, outputs), dim=-1)

        return outputs
