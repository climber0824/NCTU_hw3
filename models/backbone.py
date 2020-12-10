import torch
import torch.nn as nn
import pickle


class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        norm_layer=nn.BatchNorm2d,
        dilation=1,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(
        self,
        layers,
        atrous_layers=[],
        block=Bottleneck,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [
            m for m in self.modules() if isinstance(m, nn.Conv2d)
        ]

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    dilation=self.dilation,
                ),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.norm_layer,
                self.dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, norm_layer=self.norm_layer)
            )

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith("layer"):
                idx = int(key[5])
                new_key = "layers." + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(
        self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck
    ):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(
            block,
            conv_channels // block.expansion,
            blocks=depth,
            stride=downsample,
        )


class ResNetBackboneGN(ResNetBackbone):
    def __init__(self, layers, num_groups=32):
        super().__init__(
            layers, norm_layer=lambda x: nn.GroupNorm(num_groups, x)
        )

    def init_backbone(self, path):
        """ The path here comes from detectron. So we load it differently. """
        with open(path, "rb") as f:
            state_dict = pickle.load(
                f, encoding="latin1"
            )  # From the detectron source
            state_dict = state_dict["blobs"]

        our_state_dict_keys = list(self.state_dict().keys())
        new_state_dict = {}

        def gn_trans(x):
            return "gn_s" if x == "weight" else "gn_b"

        def layeridx2res(x):
            return "res" + str(int(x) + 2)

        def block2branch(x):
            return "branch2" + ("a", "b", "c")[int(x[-1:]) - 1]

        # Transcribe each Detectron weights name to a Yolact weights name
        for key in our_state_dict_keys:
            parts = key.split(".")
            transcribed_key = ""

            if parts[0] == "conv1":
                transcribed_key = "conv1_w"
            elif parts[0] == "bn1":
                transcribed_key = "conv1_" + gn_trans(parts[1])
            elif parts[0] == "layers":
                if int(parts[1]) >= self.num_base_layers:
                    continue

                transcribed_key = layeridx2res(parts[1])
                transcribed_key += "_" + parts[2] + "_"

                if parts[3] == "downsample":
                    transcribed_key += "branch1_"

                    if parts[4] == "0":
                        transcribed_key += "w"
                    else:
                        transcribed_key += gn_trans(parts[5])
                else:
                    transcribed_key += block2branch(parts[3]) + "_"

                    if "conv" in parts[3]:
                        transcribed_key += "w"
                    else:
                        transcribed_key += gn_trans(parts[4])

            new_state_dict[key] = torch.Tensor(state_dict[transcribed_key])

        self.load_state_dict(new_state_dict, strict=False)


def construct_backbone(cfg):
    backbone = cfg.type(*cfg.args)

    num_layers = max(cfg.selected_layers) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone
