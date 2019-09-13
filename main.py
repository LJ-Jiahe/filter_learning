import config as cfg
from tqdm import tqdm



# Setting up dataset & dataloader
train_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.train_input_dir,
    target_dir=cfg.train_target_dir,
    transform=cfg.transform)

test_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.test_input_dir,
    target_dir=cfg.test_target_dir,
    transform=cfg.transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.train_batch_size,
    shuffle=cfg.train_shuffle)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=cfg.train_batch_size,
    shuffle=cfg.test_shffule)

# Other parameteres
criterion = cfg.criterion
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# Initialize model
model = classes.


# Start training
print("\nTraining starts\n")
start_time = time.time()

for epoch in range(cfg.epoch):
    print("\nEpoch " + str(epoch) + " of " + cfg.epoch + "\n")
        
    for ite, datapoint in enumerate(tqdm(train_loader, desc='Train')):
        input_batch = data_batch['input_image']
        target_batch = data_batch['target_image']

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()

        optimizer.zero_grad()
        output_batch = model(
            



