'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-06-17 14:52:27
@Description:  
'''
import torchvision
import matplotlib.pyplot as plt 
import numpy as np 

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy() # tensor to array
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # CHW to HWC
    plt.show()

def visualize(dataloader):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')

    dataiter = iter(dataloader)
    images, labels = dataiter.next() # get batch of image tensor
    print(images.shape, labels.shape)

    imgs_grid = torchvision.utils.make_grid(images)
    print(imgs_grid.shape)

    imshow(imgs_grid)
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
