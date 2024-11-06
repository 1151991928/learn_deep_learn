import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    test_dataset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=0)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True,num_workers=0)
    test_data_iter=iter(test_dataloader)
    test_image,test_label=test_data_iter.next()



    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    model=LeNet()
    loss_function=nn.CrossEntropyLoss()
    optimzer=optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(20):
        running_loss=0.0
        for step,(inputs,labels) in enumerate(train_dataloader,start=0):
            optimzer.zero_grad()
            outputs=model(inputs)
            loss=loss_function(outputs,labels)
            loss.backward()
            optimzer.step()

            running_loss+=loss.item()
            if step % 500 == 499:    # print every 500 mini-batches 就每五百轮验证一次
                with torch.no_grad():
                    outputs = model(test_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')
    save_path = './Lenet.pth'
    torch.save(model.state_dict(), save_path)





if __name__ == '__main__':
    main()