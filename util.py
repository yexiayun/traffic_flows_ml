import torch

# 保存模型到本地
def save_checkpoint(epoch, 
                    model, 
                    lossMIN, 
                    optimizer, 
                    checkpoint_path):

    launchTimestamp = str(time.time())
    torch.save({'epoch': epoch + 1, 
                'state_dict': model.state_dict(), 
                'best_loss': lossMIN,
                'optimizer': optimizer.state_dict()
               }, 
               checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')

# 从本地加载模型
def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer

# 转换为lstm所需要的维度
def get_batchs_for_lstm(x, time_steps):
    return x.t().view(time_steps,-1,1)

def compute_val_loss(net, 
                     val_loader, 
                     loss_function, 
                     epoch, 
                     device, 
                     time_steps):
    """
    compute mean loss on validation set
    Parameters
    ----------
    net: model
    val_loader: DataLoader
    loss_function: func
    epoch: int, current epoch
    """
    val_loader_length = len(val_loader)
    tmp = []
    for index, (val_x, val_y) in enumerate(val_loader):
        output = net(get_batchs_for_lstm(val_x, time_steps))
        l = loss_function(output, val_y)  # l is a tensor, with single value
        tmp.append(l.item())
        print('validation batch %s / %s, loss: %.2f' % (
            index + 1, val_loader_length, l.item()))

    validation_loss = sum(tmp) / len(tmp)
    return validation_loss