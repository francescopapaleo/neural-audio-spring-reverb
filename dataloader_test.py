from dataloader_subset import DataLoader
from config import *


data_loader = DataLoader(DATASET)
x_train, y_train, x_valid, y_valid, x_test, y_test = data_loader.load_data()


# # Iterate over the train_dataloader
# for batch_idx, (input_data, output_data) in enumerate(train_dataloader):
#     print(f"Batch {batch_idx}:")
#     print(f"Input Data shape: {input_data.shape}")
#     print(f"Output Data shape: {output_data.shape}")
#     print()

#     # You can break the loop after the first iteration to see just one batch
#     break

# # Iterate over the val_dataloader
# for batch_idx, (input_data, output_data) in enumerate(val_dataloader):
#     print(f"Batch {batch_idx}:")
#     print(f"Input Data shape: {input_data.shape}")
#     print(f"Output Data shape: {output_data.shape}")
#     print()

#     # You can break the loop after the first iteration to see just one batch
#     break