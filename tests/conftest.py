import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.Compact import Compact as SRVGGNetCompact


def tiny_model() -> ImageModelDescriptor:
    """Randomly initialised 2x Compact net: 6 stacked 3x3 convs, so its receptive-field radius is 6 px."""
    torch.manual_seed(0)
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=4, upscale=2)
    return ModelLoader().load_from_state_dict(net.state_dict()).eval()


@pytest.fixture
def model() -> ImageModelDescriptor:
    return tiny_model()
