import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.config.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, 2)

    def forward(self, x):
        """
        The forward method defines the computation performed at every call 
        """
        batch = x.shape[0]

        # Region embedding
        x = self.conv_region_embedding(x) # It transforms the input sentences into region embeddings.       # [batch_size, channel_size, length, 1]

        # The network uses multiple sets of 2D convolution layers (conv3) and max pooling layers (pooling) for feature extraction and dimensionality reduction.
        x = self.padding_conv(x)                      
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        """
        the _block method represents the repeated convolution and pooling operations in the DPCNN architecture. 
        """
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        """
        The predict method is used for making predictions with the trained model.
        """
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels

