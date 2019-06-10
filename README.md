# Associative Convolutional Layers
As Convolutional Neural Networks (CNNs) have become deeper and/or wider to achieve higher accuracies, the number of parameters associated with their convolutional layers has become much more significant. In this work, we provide a general and straightforward method for significantly reducing the number of parameters of convolutional layers of CNNs, during both the training and inference phases, without compromising the accuracy, training time or inference time.

We introduce a simple auxiliary neural network which generates "slices" of sets of convolutional filters from a low dimensional latent space.
This simple auxiliary neural network which we call “Convolutional Slice Generator,” or CSG for short, is unique to the whole network and in this sense the convolutional layers are all associated. Each slice corresponds to a "code vector," which is its representation in the aforementioned low dimensional latent space.
During the training of the CNN, instead of training the filters of the convolutional layers directly, only the parameters of the CSG and the code vectors of the slices of the filters are trained. The model becomes even more efficient if pre-trained parameters of the CSG are reused. This results in a significant reduction in the number of trainable parameters. Also, once the training is concluded, the CNN can be fully represented using only the parameters of the CSG, the code vectors, the fully connected layers, and the architecture of the CNN. To show the capability of the proposed approach, we considered two of the widely used CNN architectures, namely ResNet and DenseNet and applied this method to them on the CIFAR10 dataset. Our experiments show that while our approach significantly reduces the number of network parameters (up to $5\times$ in our experiments and up to $18\times$ as the network grows in depth), even when applied to already compressed and efficient CNNs such as DenseNet-BC, in most cases the accuracy of the new network remains within one percent of the original network's accuracy and in some cases the accuracy slightly improves. 

This code is based on the following references: 
-- https://github.com/liuzhuang13/DenseNet 
-- https://github.com/D-X-Y/ResNeXt-DenseNet 
-- https://github.com/kuangliu/pytorch-cifar 


## Implementing The Convolutional Slice Generators
As explained in the paper, the CSG and code vectors replace the ordinary kernel parameters in our method as they are responsible for generating slices of sets of convolutional filters.
There are only three main steps that are required to implement CSG on any network:
1. Defining the CSG in the constructor of the main class of the network (remember that the CSG is unique to each network so it is defined only once).
2. Defining the code vectors in the constructor of each layer (remember that each layer has its own codes).
3. Calculating the slices and combining them to make the kernel in each forward pass of each convolutional layer.


Let us go through an example. Here we mention the main changes to DenseNet-BC as an example. 
1. 
```
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, reg_mode = False, org = False):
        super(DenseNet, self).__init__()
        ....
        #################################################
        if not org:
            #############################################
            ## Here is where the CSG is defined.
            self.CSG = torch.nn.Linear(CODE_SIZE,np.prod(SLICE_SHAPE),bias=False)
        else:
            self.CSG = None
        
    def _make_dense_layers(self, block, in_planes, nblock, reg_mode=False):
        layers = []
        for i in range(nblock):
            ####################################################
            ## We merely pass the reference of CSG to all blocks
            layers.append(block(in_planes, self.growth_rate, self.CSG, reg_mode=reg_mode))
            ....
```
        
2. 

```
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, CSG, reg_mode=False):
    
        ...
        
        ###############################################################
        #### The following covolutional layer is replaced by our CSG generated kernel
        #self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        ###############################################################
        ## Here we set things up for the CSG including defining a code matrix (set of code vectors if you wish) for each layer
        self.filter_size2 = 3
        self.in_filters2 = 4*growth_rate
        self.out_filters2 = growth_rate
        self.num_slices2 = int(np.ceil(self.in_filters2/SLICE_SHAPE[0])*np.ceil(self.out_filters2/SLICE_SHAPE[1]))
        self.code2 = torch.nn.Parameter(torch.randn([self.num_slices2]+[CODE_SIZE]))
        self.kernel2 = None
        self.kernel2_defined = False
        ...
```

3. 
```
def forward(self, x):
        ...
        #########################################
        ## Updating the kernel
        self.kernel2 = self.CSG(self.code2)
        self.kernel2 = self.kernel2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
        self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        self.kernel2_defined = True
        
        
        ###########################################
        ### This is replaced by our convolution using our CSG generated kernel
        # out = self.conv2(F.relu(self.bn2(out)))
        
        
        ###########################################
        ## Convolution with our kernel
        out = F.conv2d(F.relu(self.bn2(out)),self.kernel2,padding=1)
        ...
        
```

## Authors:
Hamed Omdivar, Vahideh Akhlaghi
