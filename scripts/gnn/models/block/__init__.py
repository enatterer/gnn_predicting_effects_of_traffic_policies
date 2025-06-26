import os
import sys

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.block.block import EIGNBlock
from gnn.models.block.laplacian_conv_block import EIGNBlockMagneticEdgeLaplacianConv
from gnn.models.block.laplacian_conv_with_node_transformation_block import (
    EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv,
)

__all__ = [
    'EIGNBlock',
    'EIGNBlockMagneticEdgeLaplacianConv',
    'EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv']
