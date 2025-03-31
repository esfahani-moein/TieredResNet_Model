import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.optim.lr_scheduler import MultiStepLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1e9:.2f} GB")
    else:
        print("No GPU available")

print_gpu_memory()

# Training hyperparameters
num_epochs = 200
batch_size = 512
learning_rate =  0.1 * (512/128)  # Scale learning rate with batch size
weight_decay = 5e-4
milestones = [60, 120, 160]  # Learning rate schedule milestones
gamma = 0.2  # Learning rate decay factor
best_acc = 0  # Best test accuracy



# Load CIFAR-10 Dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Download and load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8,  # Increased from 2
    pin_memory=True  # Add this for faster data transfer
)

# Download and load the testing set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Define classes in CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ModelNet(nn.Module):
    def __init__(self, depth=26, alpha=48, num_classes=10):
        super(ModelNet, self).__init__()
        self.inplanes = 16
    
        n = (depth - 2) // 6
        block = BasicBlock
            
        self.addrate = alpha / (3 * n * 1.0)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Layers
        self.layer1 = self._make_layer(block, n, stride=1)
        
        # Transition layers to handle channel changes
        layer1_planes = self.inplanes
        self.trans1 = nn.Sequential(
            nn.Conv2d(16, int(layer1_planes), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(layer1_planes))
        )
        
        self.layer2 = self._make_layer(block, n, stride=2)
        
        layer2_planes = self.inplanes
        self.trans2 = nn.Sequential(
            nn.Conv2d(int(layer1_planes), int(layer2_planes), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(layer2_planes))
        )
        
        self.layer3 = self._make_layer(block, n, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine feature size dynamically with a dummy forward pass
        dummy_input = torch.zeros(1, 3, 32, 32)  # CIFAR image size
        with torch.no_grad():
            dummy_output = self._forward_features(dummy_input)
            feature_size = dummy_output.size(1)
        
        # Classifier with dynamically determined feature size
        self.fc = nn.Linear(feature_size, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, blocks, stride):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True),
            )
            
        layers = []
        
        # Store the starting number of planes for this layer
        current_planes = int(self.inplanes)
        
        # Create all blocks with the same number of planes
        for i in range(blocks):
            if i == 0:
                layers.append(block(current_planes, current_planes, stride, downsample))
            else:
                layers.append(block(current_planes, current_planes))
        
        # Update planes for the next layer (all at once, after the layer is complete)
        self.inplanes += blocks * self.addrate
                
        return nn.Sequential(*layers)
    
    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.trans1(x)
        
        x = self.layer2(x)
        x = self.trans2(x)
        
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x

# Instantiate the model
depth = 110 # 110
alpha = 270 # 270

start_time = time.time()

net = ModelNet(depth=depth, alpha=alpha, num_classes=10).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(net)
# Print the model architecture
print(net)
creation_time = time.time() - start_time



print(f"ModelNet created with {num_params:,} parameters in {creation_time:.2f} seconds")
print(f"Model architecture type: ModelNet-{depth} with alpha={alpha}")

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')
    
    return train_loss/len(trainloader), 100.*correct/total

# Testing function
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'Epoch: {epoch} | Test Loss: {test_loss/len(testloader):.3f} | Test Acc: {100.*correct/total:.3f}%')
    
    return test_loss/len(testloader), 100.*correct/total

# Optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 90, 120], gamma=0.2)



checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)




# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# Arrays to store metrics
train_losses = []
train_accs = []
test_losses = []
test_accs = []

# Training function
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f'Batch: {batch_idx}/{len(trainloader)} | ' +
                  f'Loss: {train_loss/(batch_idx+1):.3f} | ' +
                  f'Acc: {100.*correct/total:.3f}% | ' +
                  f'Time: {time.time() - start_time:.1f}s')
    
    # Save metrics
    epoch_loss = train_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    print(f'Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.3f}%')
    return epoch_loss, epoch_acc

# Testing function
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = net(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Track statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Save metrics
    epoch_loss = test_loss / len(testloader)
    epoch_acc = 100. * correct / total
    test_losses.append(epoch_loss)
    test_accs.append(epoch_acc)
    
    print(f'Test Loss: {epoch_loss:.3f} | Test Acc: {epoch_acc:.3f}%')
    
    # Save checkpoint if it's the best model so far
    if epoch_acc > best_acc:
        print(f'Best accuracy: {epoch_acc:.3f}% (previous: {best_acc:.3f}%)')
        best_acc = epoch_acc
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs
        }, checkpoint_path)
        print(f'Saved best checkpoint to {checkpoint_path}')
    
    # Save regular checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
    
    return epoch_loss, epoch_acc



# Training loop with error handling
try:
    start_time = time.time()
    
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train and test
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr}')
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        
    total_time = time.time() - start_time
    print(f'Training completed in {total_time/60:.2f} minutes')
    print(f'Best accuracy: {best_acc:.3f}%')
    
    # Save final model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }, os.path.join(checkpoint_dir, 'final_model.pth'))

except Exception as e:
    print(f"Error during training: {e}")
    
    # Save emergency checkpoint
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }, os.path.join(checkpoint_dir, 'emergency_backup.pth'))
    
    import traceback
    traceback.print_exc()

