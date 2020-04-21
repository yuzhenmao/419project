import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.utils import shuffle
import torch.optim as optim
from sklearn.model_selection import train_test_split

train_file = 'training.csv'
test_file = 'test.csv'
test_type = 'IdLookupTable.csv'
MODEL_STORE_PATH = './'

# Hyperparameters
num_epochs = 6
num_classes = 10
batch_size = 20
learning_rate = 0.001


def csvFileRead(filename):
    df = pd.read_csv(filename, header=0, encoding='GBK')

    if 'train' in filename:
        df = df.dropna()
    return df


def preTrain():
    df = csvFileRead(train_file)

    df.Image = df.Image.apply(lambda im: np.fromstring(im, sep=' '))
    X = df.Image
    
    y = df[df.columns[:-1]].values
    y = (y-48)/48.
    y = y.astype(np.float32)

    yd = dict()
    for i in range(len(df.columns[:-1].values)):
        yd[df.columns[i]] = i

    return X, y, yd


def preTest():
    df = csvFileRead(test_file)

    df.Image = df.Image.apply(lambda im: np.fromstring(im, sep=' '))
    X = np.vstack(df.Image.values) / 255.
    X = X.astype(np.float32)
    X = X.reshape(-1,96,96,1)
    X = X.transpose((0, 3, 1, 2))

    df = csvFileRead(test_type)
    ImageId = df.ImageId.values - 1
    FeatureName = df.FeatureName.values

    return ImageId, FeatureName, X


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        X = self.data[idx, 0]
        X = np.vstack(X) / 255.
        X = X.astype(np.float32)
        X = X.reshape(96,96,1)
        X = X.transpose((2, 0, 1))
        y = self.data[idx, 1:]
        y = y.astype(np.float32)
        sample = {'X': X, 'y': y}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        X, y = sample['X'], sample['y']

        return {'X': torch.from_numpy(X),
                'y': torch.from_numpy(y)}

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=2),
            # output size = (W-F+2P)/S+1 = (96-9+4)/1+1 = 92
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            # (92-2)/2+1 = 46  the output Tensor for one image, will have the dimensions: (32,46,46)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, dilation=2),
            # output size = (W-F+2P)/S+1 = (46-9+4)/1+1 = 42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            # (42-2)/2+1 = 21  the output Tensor for one image, will have the dimensions: (64,21,21)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(21 * 21 * 64, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model
    
if __name__ == '__main__':
    df = csvFileRead(test_type)
    train_X, train_y, yd = preTrain()
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=33)
    ImageId, FeatureName, test_X = preTest()

    # change data to Dataset class
    trans = transforms.Compose([ToTensor()])

    train = np.column_stack((X_train, y_train))
    train_data = MyDataset(train, trans)
    vali = np.column_stack((X_test, y_test))
    vali_data = MyDataset(vali, trans)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    model = ConvNet()
    model = model.to(device)
    
    train_loader = DataLoader(train_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    vali_loader = DataLoader(vali_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    avg_loss = []
    avg_loss_1 = []
    model = model.float()
    for epoch in range(num_epochs):
        loss_list = []
        loss_list_1 = []
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            X = data['X'].to(device)
            y = data['y'].to(device)

            X = X.float()
            y = y.float()
            # Run the forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Avg.Loss: {}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss / 10))
                running_loss = 0.0
        print('Finished Training')
        avg = sum(loss_list)/len(loss_list)
        avg_loss.append(avg)

        model.eval()
        running_loss = 0.0
        total = len(vali_loader)
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
                X = data['X'].to(device)
                y = data['y'].to(device)

                X = X.float()
                y = y.float()
                # Run the forward pass
                outputs = model(X)
                loss = criterion(outputs, y)
                loss_list_1.append(loss.item())
                running_loss += loss.item()
                if (i+1)%10 == 0:
                    print('Step [{}/{}], Avg.Loss: {}'.format(i+1, total, running_loss/10))
                    running_loss = 0.0
        print('Test finished')
        avg = sum(loss_list_1)/len(loss_list_1)
        avg_loss_1.append(avg)
        
    # Save model
    checkpoint = {'model': ConvNet(),
              'state_dict': model.state_dict()}

    torch.save(checkpoint, MODEL_STORE_PATH + 'conv_net_model.pth')   

    
    # Test
    file_path = MODEL_STORE_PATH + 'conv_net_model.pth'
    model = load_checkpoint(file_path).to(device)
    test_loader = DataLoader(test_X, batch_size=10, shuffle=False, num_workers=4)
    
    prediction = np.empty([1,30])
    with torch.no_grad():
        for i,test_X in enumerate(test_loader):
            test_X = test_X.to(device)
            test_X = test_X.float()
            outputs = model(test_X)
            outputs = outputs.cpu()
            pred = np.vstack(outputs.numpy())
            prediction = np.append(prediction, pred, axis=0)
    prediction = np.delete(prediction, 0, 0)
    
    for i in range(len(FeatureName)):
        imageID = ImageId[i]
        featureID = yd[FeatureName[i]]
        df.iloc[i,3] = prediction[imageID, featureID]*48.+48.

    result = df[['RowId', 'Location']]
    result.to_csv('Result.csv', index=0)    
    print('SUCCEED')    


    f = plt.figure()
    plt.plot(avg_loss, linewidth=3, label='train')
    plt.plot(avg_loss_1, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()
    f.savefig('loss.png')
