import torch
import torch.utils.data as data
torch.manual_seed(6)

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)
torch.cat((x.view(len(x), -1), y.view(len(y), -1)), 1)

data_set = data.TensorDataset(x, y)
data_set

BATCH_SIZE = 5

loader = data.DataLoader(dataset=data_set,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=1
                         )
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print(epoch, step, batch_x.numpy(), batch_y.numpy())


import torchvision
from torchvision import datasets, transforms

img_dir = 'PyTorch-master/images'
img_data = datasets.ImageFolder(img_dir, transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]))
print(img_data.classes)
print(img_data.class_to_idx)
print(len(img_data.imgs))

loader = data.DataLoader(img_data, batch_size=3, shuffle=True, num_workers=1)
for img, label in loader:
    print(img.size())
    print(label)

train_dataset = datasets.MNIST(root='PyTorch-master/data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
image, label = train_dataset[0]
print(image.size())
print(label)

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=100,
                               shuffle=True,
                               num_workers=2)
data_iter = iter(train_loader)
images,labels = data_iter.next()
for images,labels in train_loader:
    pass
print(images.size())


## Custom DataSet

# You should build custom dataset as below.
class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# Then, you can just use prebuilt torch's data loader.
custom_dataset = CustomDataset()
train_loader = data.DataLoader(dataset=custom_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)

# ========================== Using pretrained model ========================== #
# Download and load pretrained resnet.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only top layer of the model.
for param in resnet.parameters():
    param.requires_grad = False

# Replace top layer for finetuning.
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 100)  # 100 is for example.

# For test.
images = torch.autograd.Variable(torch.randn(10, 3, 224, 224))
outputs = resnet(images)
print(outputs.size())  # (10, 100)
