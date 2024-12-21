import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader,
               model,
               loss_fn,
               optimizer,
               batch_size,
               verbose=0):

  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  train_loss = 0

  for batch, (X, y) in enumerate(dataloader):
    
    pred = model(X)
    loss = loss_fn(pred, y)

    #### YOUR CODE HERE ####
    # Compute prediction and loss. Backpropagate and update parameters.
    # Don't forget to set the gradients to zero with 
    # optimizer.zero_grad(), after each update.
    # Our implementation has 3 lines of code, but feel free to deviate from that
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    ## YOUR CODE ENDS HERE ##


  # After going through all batches and updating models parameter based on each
  # batch, we find the loss on the train dataset.
  # Evaluating the model with torch.no_grad() ensures that no gradients are
  # computed during test mode, also serves to reduce unnecessary gradient
  # computations and memory usage for tensors with requires_grad=True
  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      train_loss += loss_fn(pred, y).item()

  train_loss /= num_batches
  # print(f"Train Avg loss: {train_loss:>8f}")
  return train_loss


def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0

  # Evaluating the model with torch.no_grad() ensures that no gradients are
  # computed during test mode, also serves to reduce unnecessary gradient
  # computations and memory usage for tensors with requires_grad=True
  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      test_loss += loss_fn(pred, y).item()

  test_loss /= num_batches
  # print(f"Test  Avg loss: {test_loss:>8f}")
  return test_loss


def train(TrainDataset,
          ValDataset,
          model,
          loss_function,
          optimizer,
          max_epoch=100,
          batch_size=64):

  train_loss = torch.zeros(max_epoch)
  test_loss = torch.zeros(max_epoch)

  train_dataloader = DataLoader(dataset=TrainDataset,
                                batch_size=batch_size,
                                shuffle=True)
  test_dataloader = DataLoader(dataset=ValDataset,
                               batch_size=batch_size,
                               shuffle=True)

  for t in range(max_epoch):
    # print(f"Epoch {t+1} -------------------")

    train_loss[t] = train_loop(train_dataloader,
                               model,
                               loss_function,
                               optimizer,
                               batch_size)

    test_loss[t] = test_loop(test_dataloader,
                             model,
                             loss_function)
  # print("Done!")

  return train_loss, test_loss


##################################################
##### CODE USED TO TRAIN RNN AND LSTM MODELS #####
##################################################

###########
### RNN ### 
###########
"""
if __name__ == "__main__":
  import util
  from make_dataset import make_adding_train_val_dataset

  add10 = make_adding_train_val_dataset(train_count=10000,
                                          val_count=1000,
                                          sequence_length=10)

  add25 = make_adding_train_val_dataset(train_count=10000,
                                          val_count=1000,
                                          sequence_length=25)

  add50 = make_adding_train_val_dataset(train_count=10000,
                                          val_count=1000,
                                          sequence_length=50)
  # What should be the value of RNN_input_size and RNN_output_size?
  RNN_input_size = 2
  RNN_output_size = 1

  # Performed a grid search over the hyperparameters below
  opt_name = 'adam' # let's stick to adam.

  hs = [5, 15]
  lr = [0.1, 0.01, 0.001]
  datasets = (('10', add10), ('25', add25),('50', add50))

  best_loss = 10000000
  best_model = None
  for seq_len,(Add10Trainset,Add10Valset) in datasets:
    for h in hs:
      for l in lr:
        RNN_hidden_size = h # choose from 5, or 15
        learning_rate = l # choose from 0.1, 0.01, or 0.001

        RNNmodel10, train_loss10, val_loss10 = \
          train1LayerVanillaRNN(Add10Trainset,
                                          Add10Valset,
                                          RNN_input_size,
                                          RNN_output_size,
                                          RNN_hidden_size,
                                          optimizer_name=opt_name,
                                          lr=learning_rate,
                                          batch_size=64,
                                          max_epoch=50)


        # Plot the loss value for train/test set for each epoch
        util.plot_loss(train_loss10,
                  val_loss10,
                  sequence_len=int(seq_len),
                  hidden_size=RNN_hidden_size,
                  lr=learning_rate,
                  model_type='rnn')

        if val_loss10.sum() < best_loss:
          best_loss = val_loss10.sum()
          best_model = (learning_rate, RNN_hidden_size)
    print(f'Finished Sequence Length {seq_len}')
    print(f'the best model is lr: {best_model[0]} and hidden_size: {best_model[1]}\n The best loss is {best_loss}')
"""
############
### LSTM ### 
############

"""
if __name__ == "__main__":
  import util
  from make_dataset import make_adding_train_val_dataset

  add10 = make_adding_train_val_dataset(train_count=10000,
                                          val_count=1000,
                                          sequence_length=10)

  add25 = make_adding_train_val_dataset(train_count=10000,
                                          val_count=1000,
                                          sequence_length=25)

  add50 = make_adding_train_val_dataset(train_count=10000,
                                          val_count=1000,
                                          sequence_length=50)
  # What should be the value of LSTM_input_size and LSTM_output_size?
  LSTM_input_size = 2
  LSTM_output_size = 1

  # Performed a grid search over the hyperparameters below
  hs = [2,5]
  lr = [0.1, 0.01, 0.001]
  datasets = (('10', add10), ('25', add25),('50', add50))
  best_losses = []
  for seq_len, (Add25Trainset,Add25Valset) in datasets:
    best_model = None
    best_loss = 100000
    for l in lr:
      for h in hs:
        LSTM_hidden_size = h # choose from 2 or 5
        opt_name = 'adam' # choose from 'sgd' or 'adam'
        learning_rate = l # choose from 0.1, 0.01, or 0.001

        LSTM_model25, lstm_train_loss25, lstm_val_loss25 = \
          train1LayerLSTM(Add25Trainset,
                                          Add25Valset,
                                          LSTM_input_size,
                                          LSTM_output_size,
                                          LSTM_hidden_size,
                                          optimizer_name=opt_name,
                                          lr=learning_rate,
                                          max_epoch=50,
                                          batch_size=64)

        # Plot the loss value for train/test set for each epoch
        util.plot_loss(lstm_train_loss25,
                  lstm_val_loss25,
                  sequence_len=int(seq_len),
                  hidden_size=LSTM_hidden_size,
                  lr=learning_rate,
                  model_type='lstm')
        
        cur_loss = lstm_val_loss25.sum()
        if cur_loss < best_loss:
          best_loss = cur_loss
          best_model = (learning_rate, LSTM_hidden_size)
    best_losses.append((best_loss, best_model))
    print(f'Finished Sequence Length {seq_len}')
    print(f'the best model is lr: {best_model[0]} and hidden_size: {best_model[1]}\n The best loss is {best_loss}')

  print(best_losses)
"""