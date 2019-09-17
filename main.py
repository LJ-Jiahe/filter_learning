import config as cfg
import classes
import torch
from tqdm import tqdm
import time
import os
import functions


# Setting up dataset & dataloader
train_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.train_gs_dir,
    target_dir=cfg.train_gs_filtered_dir,
    transform=cfg.transform)

test_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.test_gs_dir,
    target_dir=cfg.test_gs_filtered_dir,
    transform=cfg.transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.train_batch_size,
    shuffle=cfg.train_shuffle)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=cfg.test_batch_size,
    shuffle=cfg.test_shuffle)

# Initialize model
model = classes.One_Filter_Net()

# Other parameteres
criterion = cfg.criterion
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)



print("\nTraining starts\n")
start_time = time.time()

for epoch in range(int(cfg.epoch)):
    print("\nEpoch " + str(epoch) + " of " + str(cfg.epoch) + "\n")

#  Test
    test_loss_total = 0
    for test_ite, test_datapoint in enumerate(tqdm(test_loader, desc='Test')):
        test_input_batch = test_datapoint['input_image'].type(torch.FloatTensor)
        test_target_batch = test_datapoint['target_image'].type(torch.FloatTensor)

        test_input_batch = test_input_batch[:, 0, :, :].unsqueeze_(1)
        test_target_batch = test_target_batch[:, 0, :, :].unsqueeze_(1)
    
        if torch.cuda.is_available():
            test_input_batch = test_input_batch.cuda()
            test_target_batch = test_target_batch.cuda()

        test_output_batch = model(test_input_batch)

        test_loss = criterion(test_output_batch, test_target_batch)
        test_loss_total += test_loss.item()

    test_loss_avg = test_loss_total / test_loader.__len__()
    test_loss_loc = os.path.join(cfg.loss_folder, 'test_loss')
    functions.append_to_pickle_file(test_loss_loc, [epoch, test_loss_avg])

# Start training
    train_loss_total = 0
    for train_ite, train_datapoint in enumerate(tqdm(train_loader, desc='Train')):
        #typecasting to FloatTensor as it is compatible with CUDA
        train_input_batch = train_datapoint['input_image'].type(torch.FloatTensor)
        train_target_batch = train_datapoint['target_image'].type(torch.FloatTensor)

        train_input_batch = train_input_batch[:, 0, :, :].unsqueeze_(1)
        train_target_batch = train_target_batch[:, 0, :, :].unsqueeze_(1)

        if torch.cuda.is_available():
            train_input_batch = train_input_batch.cuda()
            train_target_batch = train_target_batch.cuda()

        optimizer.zero_grad()
        train_output_batch = model(train_input_batch)
        train_loss = criterion(train_output_batch, train_target_batch)
        train_loss_total += train_loss.item()
        train_loss.backward()
        optimizer.step()
        
    # # Write average loss value to file once every epoch
    train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
    train_loss_avg = train_loss_total / train_loader.__len__()
    functions.append_to_pickle_file(train_loss_loc, [epoch, train_loss_avg])
    
 # Print Loss
    time_since_start = (time.time()-start_time) / 60
    print('\nEpoch: {} \nLoss avg: {} \nTest Loss avg: {} \nTime(mins) {}'.format(
         epoch, train_loss_avg, test_loss_avg, time_since_start))

# Save model
    if epoch % cfg.save_epoch == 0:
        ckpt_folder = os.path.join(cfg.ckpt_folder, 'model_epoch_' + str(epoch) + '.pt')
        torch.save(model, ckpt_folder)
        print("\nmodel saved at epoch : " + str(epoch) + "\n")
