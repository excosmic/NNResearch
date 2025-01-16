import torch as tc
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision as tcv
import torchvision.transforms as transforms



class Dilution(nn.Module):
    def __init__(self, from_size:list[int], to_size:list[int]):
        def dilution(inputTensor:tc.Tensor, fromSize:list[int], toSize:list[int]) -> tc.Tensor:
            trans:transforms.Compose = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(fromSize),
                transforms.ToTensor()
            ])
            inputTensor = trans(inputTensor)
            ans:tc.Tensor = tc.zeros(inputTensor.size(0), inputTensor.size(1), toSize[0], toSize[1]).to(inputTensor.device)
            offset:list[int] = [0, 0]
            for y in range(fromSize[0]):
                for x in range(fromSize[1]):
                    # TODO: offset parameters should be enable here, but it is not implemented yet.
                    ans[:, :, int(y*toSize[0]/fromSize[0]+offset[0]), int(x*toSize[1]/fromSize[1]+offset[1])] = inputTensor[:, :, y, x]
            return ans
        self.dilution = dilution
        self.fromSize = from_size
        self.toSize = to_size
        pass
    
    def __call__(self, x:tc.Tensor):
        return self.dilution(x, self.fromSize, self.toSize)
    pass
# TODO: Test the Dilution class
class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (3, 3), stride=1, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding="same")
        self.conv2 = nn.Conv2d(6, 16, (3, 3), stride=1, padding="same")
        self.conv3 = nn.Conv2d(16, 6, (3, 3), stride=1, padding="same")
        return
        
    def forward(self, x):
        # TODO
        return
    pass
print(BinaryNet())