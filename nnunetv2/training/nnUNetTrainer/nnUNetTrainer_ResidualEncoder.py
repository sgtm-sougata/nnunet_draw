import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetTrainer_ResidualEncoder(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # Override the default PlainConvUNet with ResidualEncoderUNet
        model = ResidualEncoderUNet(
            input_channels=num_input_channels,
            n_stages=len(configuration_manager.conv_kernel_sizes),
            features_per_stage=configuration_manager.UNet_base_num_features,
            conv_op=nn.Conv3d,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_blocks_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=dataset_json['labels'].__len__(),
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=enable_deep_supervision
        )
        return model
