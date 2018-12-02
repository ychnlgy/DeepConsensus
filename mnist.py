#!/usr/bin/python3

import numpy, torch, torchvision, random, tqdm, os

DIR = os.path.dirname(__file__)
ROOT = os.path.join(DIR, "data")

def get_mnist(download=0):
    
    download = int(download)
    
    NUM_CLASSES = 10
    CHANNELS = 1
    IMAGESIZE = (32, 32)
    
    train = torchvision.datasets.MNIST(root=ROOT, train=True, download=download)
    trainData = train.train_data.view(-1, 1, 28, 28).float()/255.0
    trainData = convert_size(trainData, IMAGESIZE)
    trainLabels = torch.LongTensor(train.train_labels)
    
    test = torchvision.datasets.MNIST(root=ROOT, train=False, download=download)
    testData = test.test_data.view(-1, 1, 28, 28).float()/255.0
    testData = convert_size(testData, IMAGESIZE)
    testLabels = torch.LongTensor(test.test_labels)

    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnist_corrupt(download=0, **kwargs):
    return make_corrupt(get_mnist(download), **kwargs)

def get_mnist64(download=0):
    IMAGESIZE = (64, 64)
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, _ = get_mnist(download)
    trainData = convert_size(trainData, IMAGESIZE)
    testData = convert_size(testData, IMAGESIZE)
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def get_mnist64_corrupt(download=0, **kwargs):
    return make_corrupt(get_mnist64(download), **kwargs)

def create_trainvalid_split(p, train_dat, train_lab, test_dat, test_lab, trainbatch, testbatch):
    n = len(train_dat)
    indices = numpy.arange(n)
    numpy.random.shuffle(indices)
    split = int(round(p*n))
    trainidx = torch.from_numpy(indices[split:n])
    valididx = torch.from_numpy(indices[:split])
    
    dataloader = create_loader(train_dat[trainidx], train_lab[trainidx], trainbatch)
    validloader = create_loader(train_dat[valididx], train_lab[valididx], testbatch)
    testloader = create_loader(test_dat, test_lab, testbatch)
    
    return dataloader, validloader, testloader

# === Helpers === 

def create_loader(dat, lab, batch):
    dataset = torch.utils.data.TensorDataset(dat, lab)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return dataloader

def convert_size(data, size):
    N, C, W, H = data.size()
    X, Y = size
    CX = (X - W)//2
    CY = (Y - H)//2
    out = torch.zeros(N, C, *size)
    out[:,:,CX:CX+W,CY:CY+H] = data
    return out

def make_corrupt(original, corrupt_train=False, corrupt_test=True, **kwargs):
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE = original
    
    if int(corrupt_train):
        trainData = make_data_corrupt(trainData, kwargs)
    if int(corrupt_test):
        testData = make_data_corrupt(testData, kwargs)
    
    return trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE

def make_data_corrupt(data, kwargs):
    N, C, W, H = data.size()
    out = [
        translate(
            im.permute(1, 2, 0).squeeze().numpy(),
            **kwargs
        ) for im in tqdm.tqdm(data, desc="Corrupting test set", ncols=80)
    ]
    out = numpy.array(out)
    out = torch.from_numpy(out).float()
    out = out.view(N, W, H, C).permute(0, 3, 1, 2)
    return out

def translate(im, mintrans=0, maxtrans=0):
    
    mintrans = int(mintrans)
    maxtrans = int(maxtrans)

    w, h = im.shape[:2]
    ox, oy = w//2, h//2
    px = random.randint(mintrans, maxtrans)
    py = random.randint(mintrans, maxtrans)

    out = numpy.zeros(im.shape)
    xa = max(0, px)
    xb = min(w, px+w)
    ya = max(0, py)
    yb = min(h, py+h)
    dx = xb - xa
    dy = yb - ya
    sx = xa - px
    sy = ya - py
    out[xa:xa+dx, ya:ya+dy] = im[sx:sx+dx, sy:sy+dy]
    
    return out

if __name__ == "__main__":
    
    from matplotlib import pyplot
    
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE = get_mnist64_corrupt(
        download = 1,
        mintrans = 20,
        maxtrans = 20
    )
    
    BATCHSIZE = 2
    
    dataloader, validloader, testloader = create_trainvalid_split(
        0.2,
        trainData,
        trainLabels,
        testData,
        testLabels,
        trainbatch=BATCHSIZE,
        testbatch=BATCHSIZE
    )
    
    def stat(loader, name):
        print("%s set has %d total elements" % (name, len(loader)*BATCHSIZE))
    
    stat(dataloader, "Training")
    stat(validloader, "Validation")
    stat(testloader, "Test")
    
    def show(loader, name):
        for batch, y in loader:
            for i, img in enumerate(batch.squeeze(), 1):
                pyplot.imshow(img.numpy(), cmap="gray")
                pyplot.title("%s set, example %d, class %d" % (name, i, y[i-1].item()))
                pyplot.show()
                pyplot.clf()
            break
    
    show(dataloader, "Training")
    show(validloader, "Validation")
    show(testloader, "Test")
