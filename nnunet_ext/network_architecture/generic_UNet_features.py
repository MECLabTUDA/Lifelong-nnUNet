import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet

class genericUNet_features(Generic_UNet):
    def forward(self, x):
        skips = []
        localizations = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            localizations.append(x)
        return localizations[-1]
