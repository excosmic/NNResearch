import torchvision as tcv
import torch as tc
import torchvision.transforms as transforms
import torch.nn as nn

# TODO: In development
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
