import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model=LeNet()
    model.load_state_dict(torch.load('Lenet.pth'))
    img=Image.open('1.jpg')
    img=transform(img)
    img=torch.unsqueeze(img,dim=0)
    with torch.no_grad():
        outputs=model(img)
        predict=torch.max(outputs,dim=1)[1].numpy()
    print(classes[int(predict)])

if __name__ == '__main__':
    main()