#!/usr/bin/python3

if __name__ == "__main__":
    
    import torch, tqdm
    
    import mnist, cnn_example, deepconsensus_example
    
    print('''
***

Training on MNIST centered in a 64x64 black image
while testing on a perturbed version of its test set,
where images are translated 20 pixels in both the x and y axes.

***
    ''')
    
    # STEP 1: Get data
    
    TRANSLATE = 20
    
    trainData, trainLabels, testData, testLabels, NUM_CLASSES, CHANNELS, IMAGESIZE = mnist.get_mnist64_corrupt(
        download = True,
        mintrans = TRANSLATE,
        maxtrans = TRANSLATE
    )
    
    BATCHSIZE = 100
    TRAIN_VS_VALIDATION_SPLIT = 0.2
    
    dataloader, validloader, testloader = mnist.create_trainvalid_split(
        TRAIN_VS_VALIDATION_SPLIT,
        trainData,
        trainLabels,
        testData,
        testLabels,
        trainbatch=BATCHSIZE,
        testbatch=BATCHSIZE
    )
    
    # STEP 2: Create models
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cnn = cnn_example.Cnn(CHANNELS, NUM_CLASSES, IMAGESIZE).to(DEVICE)
    deepconsensus_cnn = deepconsensus_example.DeepConsensusCnn(CHANNELS, NUM_CLASSES, IMAGESIZE).to(DEVICE)
    
    def count_parameters(model, name):
        count = sum(map(torch.numel, model.parameters()))
        print("%s model has %d (%.2f million) parameters." % (name, count, count/1e6))
    
    count_parameters(cnn, "CNN")
    count_parameters(deepconsensus_cnn, "DeepConsensus-CNN")
    
    # STEP 3: Train models
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    cnn_optimizer = torch.optim.Adam(cnn.parameters())
    deepconsensus_cnn_optimizer = torch.optim.Adam(deepconsensus_cnn.parameters())
    
    cnn_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer)
    deepconsensus_cnn_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(deepconsensus_cnn_optimizer)
    
    def train(model, X, y, optimizer=None):
        yh = model(X)
        loss = loss_function(yh, y)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss = loss.item()
        error = (yh.max(dim=1)[1] == y).float().mean()
        return loss, error
    
    def report(title, c_loss, d_loss, c_err, d_err, n):
        print('''\
 === %s results ===
                    CNN        DeepConsensus-CNN
Cross entropy loss: %.5f    %.5f
Accuracy          : %.5f    %.5f''' % (title, c_loss/n, d_loss/n, c_err/n, d_err/n))

    EPOCHS = 30
    
    for epoch in range(1, EPOCHS+1):
        
        cnn.train()
        deepconsensus_cnn.train()
        
        # === TRAINING ===
        
        c_loss = d_loss = c_err = d_err = n = 0.0
        
        for X, y in tqdm.tqdm(dataloader, desc="Epoch %d" % epoch, ncols=80):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            
            cl, ce = train(cnn, X, y, cnn_optimizer)
            c_loss += cl
            c_err += ce
            
            dl, de = train(deepconsensus_cnn, X, y, deepconsensus_cnn_optimizer)
            d_loss += dl
            d_err += de
            
            n += 1.0
        
        report("Training", c_loss, d_loss, c_err, d_err, n)
        
        with torch.no_grad():
            cnn.eval()
            deepconsensus_cnn.eval()
            
            # === VALIDATION ===
            
            c_loss = d_loss = c_err = d_err = n = 0.0
            
            for X, y in validloader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                
                cl, ce = train(cnn, X, y)
                c_loss += cl
                c_err += ce
                
                dl, de = train(deepconsensus_cnn, X, y)
                d_loss += dl
                d_err += de
                
                n += 1.0
            
            cnn_lr_scheduler.step(c_err/n)
            deepconsensus_cnn_lr_scheduler.step(d_err/n)
            
            report("Validation", c_loss, d_loss, c_err, d_err, n)
            
            # === TESTING ===
            
            c_loss = d_loss = c_err = d_err = n = 0.0
            
            for X, y in testloader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                
                cl, ce = train(cnn, X, y)
                c_loss += cl
                c_err += ce
                
                dl, de = train(deepconsensus_cnn, X, y)
                d_loss += dl
                d_err += de
                
                n += 1.0
            
            cnn_lr_scheduler.step(c_err/n)
            deepconsensus_cnn_lr_scheduler.step(d_err/n)
            
            report("Testing", c_loss, d_loss, c_err, d_err, n)
            
