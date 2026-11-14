"""Model definitions for PM25Vision."""

from src.models.baselines import (
    HazeRidgeBaseline,
    ImageBaseline,
    MedianBaseline,
    MetadataMLP,
    build_mobilenet_v3_small_pretrained,
    build_resnet18_pretrained,
    build_resnet18_scratch,
)
from src.models.dcm import (
    BACKBONE_SPECS,
    BackboneTapWrapper,
    DimensionConsistencyModule,
    LightTransformerEncoder,
    PFRMLHead,
    PFRMLModel,
    PFRMLOutput,
    PrototypeClassificationHead,
    PrototypeGuidedRegressionHead,
    SingleFeatureAdapter,
)
from src.models.mdnet import (
    LayerNorm2d,
    MDBlock,
    MDNet,
    MobileNetV3MDNeck,
    build_mdnet_t,
    build_mobilenet_v3_small_md_neck,
    compute_mdnet_receptive_field,
)
from src.models.pfrml_config import (
    PFRMLFullConfig,
    PFRMLHeadConfig,
    build_pfrml_from_config,
    load_pfrml_config,
)
from src.models.physics_gate import PhysicsPriorGate
from src.models.pooling import AttentionPooling

__all__ = [
    "MedianBaseline",
    "HazeRidgeBaseline",
    "MetadataMLP",
    "ImageBaseline",
    "build_resnet18_scratch",
    "build_resnet18_pretrained",
    "build_mobilenet_v3_small_pretrained",
    "LayerNorm2d",
    "MDBlock",
    "MDNet",
    "MobileNetV3MDNeck",
    "build_mdnet_t",
    "build_mobilenet_v3_small_md_neck",
    "compute_mdnet_receptive_field",
    "AttentionPooling",
    "PhysicsPriorGate",
    "BACKBONE_SPECS",
    "DimensionConsistencyModule",
    "SingleFeatureAdapter",
    "LightTransformerEncoder",
    "PrototypeClassificationHead",
    "PrototypeGuidedRegressionHead",
    "PFRMLOutput",
    "PFRMLHead",
    "BackboneTapWrapper",
    "PFRMLModel",
    "PFRMLHeadConfig",
    "PFRMLFullConfig",
    "load_pfrml_config",
    "build_pfrml_from_config",
]
