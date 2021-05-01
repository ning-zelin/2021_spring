import time
def Train(config, train_loader, val_loader, val_set, model=None):
    # create model, define a loss function, and optimizer
    if not model:
        device = get_device()
        model = Classifier().to(device)
        # # Initializer
        # initializer = XavierInitializer()
        # model.apply(initializer)
        
#    print(model)
#     print(config)
    
    # get device 
    device = get_device()
#     print(f'DEVICE: {device}')
    
    # Loss
    if config['LabelSmoothingLoss']:
        criterion = LabelSmoothingLoss(config['num_class'], smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=config['weights']) 
    
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    if config['Lookahead']:
        optimizer = Lookahead(optimizer)
    
    scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler'])(optimizer, **config['lr_scheduler_paras']) # learning rate scheduler
    
    # start training
    start = time.time()
#    dt_string = start.strftime("%Y/%m/%d %H:%M:%S")
#    print('Strat training at:', dt_string)

    # the path where checkpoint saved
    model_path = r'./model.ckpt'

    early_stop_cnt = 0
    ensemble_cnt = 0
    best_acc = 0.0
    
    lr = config['optim_hparas']['lr']
    for epoch in range(config['num_epoch']):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()
        
        train_acc /= len(train_set)
        train_loss /= len(train_loader)
        
        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels) 
                    _, val_pred = torch.max(outputs, 1) 

                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += batch_loss.item()

                scheduler.step(val_loss)
                # scheduler.step()
                
                val_acc /= len(val_set)
                val_loss /= len(val_loader)
                
                if epoch % 5 == 0:
                    print('[{:04d}/{:04d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                        epoch, 
                        config['num_epoch'], 
                        train_acc, 
                        train_loss, 
                        val_acc, 
                        val_loss
                    ))
                    
                curr_lr = optimizer.param_groups[0]['lr']
                if curr_lr != lr:
                    print('    Current learning rate: {:.8f} | early stop cnt: {}'.format(curr_lr, early_stop_cnt))
                    lr = curr_lr
                
                total_acc = val_acc * config['loss_ratio'] + train_acc * (1 - config['loss_ratio'])
                acc_check_list = [val_acc, train_acc]
                
                # if the model improves, save a checkpoint at this epoch
                if total_acc > best_acc:
                    best_acc = total_acc
                    early_stop_cnt = 0
                    torch.save(model.state_dict(), model_path)
                    print('    Saving model with Train Acc: {:.6f} | val acc: {:.6f} | total acc: {:.6f}'.format(
                        train_acc, 
                        val_acc, 
                        total_acc
                    ))
                    print()

                else:
                    early_stop_cnt += 1
                
                if stop_criterion(early_stop_cnt=early_stop_cnt, curr_lr=curr_lr, acc=(train_acc, val_acc)):
                    # Stop training when satisfying stop criterion
                    print('Early stop at {} epoch'.format(epoch+1))
                    break

        else:
            print('[{:04d}/{:04d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch, config['num_epoch'], train_acc, train_loss
            ))

  
    print('learning rate is', optimizer.param_groups[0]['lr'])
    print('Ensemble_cnt: {}'.format(ensemble_cnt))
    end = time.time()
    cost_time = end - start
    print('Cost Time:', str(cost_time))
    return best_acc