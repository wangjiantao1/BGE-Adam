from torch import nn
import torch
class Resnet18(nn.Module):
    # def __init__(
    #         self,
    #         input_size: int = 784,
    #         lin1_size: int = 256,
    #         lin2_size: int = 256,
    #         lin3_size: int = 256,
    #         output_size: int = 10,
    # ) -> None:
    #     """Initialize a `SimpleDenseNet` module.
    #
    #     :param input_size: The number of input features.
    #     :param lin1_size: The number of output features of the first linear layer.
    #     :param lin2_size: The number of output features of the second linear layer.
    #     :param lin3_size: The number of output features of the third linear layer.
    #     :param output_size: The number of output features of the final linear layer.
    #     """
    #     super().__init__()
    #
    #     self.model = nn.Sequential(
    #         nn.Linear(input_size, lin1_size),
    #         nn.BatchNorm1d(lin1_size),
    #         nn.ReLU(),
    #         nn.Linear(lin1_size, lin2_size),
    #         nn.BatchNorm1d(lin2_size),
    #         nn.ReLU(),
    #         nn.Linear(lin2_size, lin3_size),
    #         nn.BatchNorm1d(lin3_size),
    #         nn.ReLU(),
    #         nn.Linear(lin3_size, output_size),
    #     )
    #
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Perform a single forward pass through the network.
    #
    #     :param x: The input tensor.
    #     :return: A tensor of predictions.
    #     """
    #     batch_size, channels, width, height = x.size()
    #
    #     # (batch, 1, width, height) -> (batch, 1*width*height)
    #     x = x.view(batch_size, -1)
    #
    #     return self.model(x)
    def __init__(self):
        super().__init__()
        # b,3,32,32
        self.l1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.l2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5)
        self.l4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l5 = nn.Dropout2d(p=0.1)
        self.l6 = nn.AdaptiveMaxPool2d((1, 1))
        self.l7 = nn.Flatten()
        self.l8 = nn.Linear(64, 32)
        self.l9 = nn.ReLU()
        self.l10 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        return x

if __name__ == "__main__":
    from torchviz import make_dot
    import torch
    model = Resnet18()
    print(model)

    # model = Backbone()
    # print(model)

