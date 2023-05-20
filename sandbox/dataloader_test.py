from data_load import DataRetriever, SubsetGenerator, SubsetRetriever
from config import *

# This script tests the classes in dataloader.py

def test_classes():
    # 1. Test DataRetriever class
    data_folder = LOCAL  # Substitute with your actual folder path
    retriever = DataRetriever(data_folder)
    x_train, y_train, x_val_test, y_val_test = retriever.retrieve_data()
    assert x_train is not None and y_train is not None and x_val_test is not None and y_val_test is not None
    print('DataRetriever Test Passed')

    # 2. Test SubsetGenerator class
    subset_size = 18  # Substitute with the desired subset size
    subset_generator = SubsetGenerator(data_folder, subset_size)
    x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset = subset_generator.retrieve_subset()
    assert len(x_train_subset) == subset_size and len(y_train_subset) == subset_size
    assert len(x_val_test_subset) == subset_size // 2 and len(y_val_test_subset) == subset_size // 2
    print('SubsetGenerator Test Passed')

    # 3. Test SubsetRetriever classn
    subset_data_folder = 'dataset_subset'
    subset_retriever = SubsetRetriever(subset_data_folder)
    x_train_concat, y_train_concat, x_val_test_concat, y_val_test_concat = subset_retriever.retrieve_data(concatenate=True)
    assert x_train_concat.shape[1] == len(x_train_subset) and y_train_concat.shape[1] == len(y_train_subset)
    assert x_val_test_concat.shape[1] == len(x_val_test_subset) and y_val_test_concat.shape[1] == len(y_val_test_subset)
    print('SubsetRetriever with Concatenation Test Passed')

    x_train_no_concat, y_train_no_concat, x_val_test_no_concat, y_val_test_no_concat = subset_retriever.retrieve_data(concatenate=False)
    assert x_train_no_concat.shape[0] == len(x_train_subset) and y_train_no_concat.shape[0] == len(y_train_subset)
    assert x_val_test_no_concat.shape[0] == len(x_val_test_subset) and y_val_test_no_concat.shape[0] == len(y_val_test_subset)
    print('SubsetRetriever without Concatenation Test Passed')

# Run the tests
# test_classes()

# Load a subset of the data
subset_retriever = SubsetRetriever('dataset_subset')
x_train, y_train, x_test, y_test = subset_retriever.retrieve_data(concatenate=True)

print(f'x', x_train.shape)
print(f'y', y_train.shape)
print()
print(f'x', x_test.shape)
print(f'y', y_test.shape)

# Load tensors

