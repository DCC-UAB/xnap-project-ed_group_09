import wandb
import time
import torch
def train_model_mse(model, dataloaders, criterion, optimizer, num_epochs=5,name_project=None,name_run=None):
    wandb.init(project=name_project,name=name_run)
    since = time.time()

    losses = {"train": [], "val": []}
    best_loss=10000000000

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.float()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                  if phase=='train':
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    outputs = outputs.float()
                    labels = labels.view(-1, 1)  # Cambiar la forma de las etiquetas a [128, 1]

                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())
                        
                    # backward + optimize only if in training phase
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                  else:
                    outputs = model(inputs)
                    outputs = outputs.float()
                    labels = labels.view(-1, 1)  # Cambiar la forma de las etiquetas a [128, 1]

                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())
                    #print(correct_num)


                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print(torch.sum(preds==labels.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            wandb.log({f"{phase}_loss": epoch_loss})



            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses