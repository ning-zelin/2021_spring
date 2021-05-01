import time
import torch
import torch.nn as nn
# 可以直接用的回归模型

config = {
    'seed': 1,
    'num_class': 39,               # number of class
    'num_epoch': 2048,             # number of training epoch
    'Lookahead': False,             # Using Lookahead
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 3e-4,                 # learning rate 
#         'weight_decay' : 1e-4,          # weight_decay
        # 'weight_decay' : 0,          # weight_decay for PReLU
        'momentum': 0.9,               # momentum for SGD
        'nesterov': True,            # nesterov for SGD
    },
    'EARLY_STOP' : 128,              # early stop setting ( = lr_param * 4)
    'lr_scheduler': 'ReduceLROnPlateau', # learning rate scheduler
#    'lr_scheduler': 'CosineAnnealingLR',
    'lr_scheduler_paras': {
        'patience' : 16,                # patience for ReduceLROnPlateau
        'factor': 0.5,                 # Reduction factor for ReduceLROnPlateau
#         'T_max': 64,                # T_max for CosineAnnealingLR
    },
#     'weights': weights,             # weights for Corss Entropy Loss
    'final_process': True,          # Post processing after ensemble
    'LabelSmoothingLoss': False        # Label Smoothing Loss
}

def Train(config, train_loader, val_loader, model, lr_shedule = True):
    min_loss = 30
    print(len(train_set))
    # create model, define a loss function, and optimizer
    
    print(model)
#     print(config)
    
    # get device 
    device = get_device()
    print(f'DEVICE: {device}')
    
    # Loss
    criterion = nn.L1Loss() 

    
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    if config['Lookahead']:
        optimizer = Lookahead(optimizer)
    
    scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler'])(optimizer, **config['lr_scheduler_paras']) # learning rate scheduler
    
    # start training
    start = time.time()
    
    # the path where checkpoint saved
    model_path = r'./model.ckpt'

    early_stop_cnt = 0

    
    lr = config['optim_hparas']['lr']
    for epoch in range(config['num_epoch']):
        train_loss = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            batch_loss = criterion(outputs, labels)
            batch_loss.backward() 
            optimizer.step() 

            train_loss += batch_loss.item()*len(labels)
        
        train_loss /= len(train_set)
        
        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels) 

                    val_loss += batch_loss.item() * len(labels)

                scheduler.step(val_loss)
#                 scheduler.step()
                

                val_loss /= len(val_set)
                
                if epoch % 5 == 0:
                    print('[{:04d}/{:04d}] Train Loss: {:3.6f} | Val loss: {:3.6f}'.format(
                        epoch, 
                        config['num_epoch'], 
                        train_loss, 
                        val_loss
                    ))

                    
               
                if lr_shedule == True:
                    curr_lr = optimizer.param_groups[0]['lr']
                    if curr_lr != lr:
                        print('    Current learning rate: {:.8f} | early stop cnt: {}'.format(curr_lr, early_stop_cnt))
                        lr = curr_lr
                    
                
                # if the model improves, save a checkpoint at this epoch
                if min_loss > val_loss:
                    min_loss = val_loss
                    early_stop_cnt = 0
                    torch.save(model.state_dict(), model_path)
                    print('    Saving model with Val_loss: {:.6f} | Train_loss {:.6f}'.format(
                        val_loss, 
                        train_loss, 
                    ))
                    print()

                else:
                    early_stop_cnt += 1

            if early_stop_cnt >= config['EARLY_STOP']:
                if epoch > 5:
                    print('Stop : reach early stop mum')
                    break
                else:
                    print('step{}, Start again now'.format(config['EARLY_STOP']))
                    early_stop_cnt = 0

        else:
            print('fail to find validation loader')

  
    end = time.time()
    cost_time = end - start
    print('Cost Time:', str(cost_time))
    return min_loss