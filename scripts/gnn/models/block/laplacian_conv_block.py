"""Block that uses the Mangnetic Edge Laplacian as graph shift operator."""

import os
import sys
import torch.nn as nn

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.conv import MagneticEdgeLaplacianConv
from gnn.models.block import EIGNBlock

class EIGNBlockMagneticEdgeLaplacianConv(EIGNBlock):
    r"""Block within the EIGN architecture that models signed (orientation signedvariant) and unsigned (orientation unsignedariant) modalities using the Magnetic Edge Laplacian as graph shift operator."""

    def initialize_convolution(
        self,
        in_channels: int,
        out_channels: int,
        signed_in: bool,
        signed_out: bool,
        **kwargs,
    ) -> nn.Module:
        return MagneticEdgeLaplacianConv(
            in_channels=in_channels,
            out_channels=out_channels,
            signed_in=signed_in,
            signed_out=signed_out,
            **kwargs,
        )
