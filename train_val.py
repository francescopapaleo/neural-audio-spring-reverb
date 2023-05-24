import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.dataload import PlateSpringDataset
from tcn import TCN
from utils.plot import plot_loss_function
import numpy as np

# Fix the random seed for reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

args = parser.parse_args()
results_dir = Path(args.results_dir)

# Creating training and validation datasets
train_dataset = PlateSpringDataset(args.dataset_dir, args.ir_dir, args.max_len, train=True)
val_dataset = PlateSpringDataset(args.dataset_dir, args.ir_dir, args.max_len, train=False)

# DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Define the model, loss function and optimizer
model = TCN(args.ninputs, args.noutputs, args.nblocks, args.nlayers, args.kernel_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Initialize variables for storing best validation loss and best model weights
best_val_loss = np.inf
best_model_weights = copy.deepcopy(model.state_dict())

loss_tracker = []
for epoch in range(args.epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        loss_tracker.append(loss.item())
        print(f'Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item()}')

    # Validation after each epoch
    with torch.no_grad():
        val_losses = []
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f'Average validation loss after epoch {epoch+1}: {avg_val_loss}')

        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            print('Best validation loss improved, saving model...')

    # Plot the loss function
    plot_loss_function(loss_tracker, args)

# Save the model with the best weights
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), f'{args.results_dir}/model_best.pt')

print('Finished Training')
