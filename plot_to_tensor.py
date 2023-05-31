import matplotlib.pyplot as plt
import io
import torchvision.transforms as transforms

def plot_to_tensor():
    plt.figure()
    plt.hist(param_np, bins='auto')  # arguments are passed to np.histogram
    plt.title(name)
    plt.close()

    # Convert plot to tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    return image

# In your training loop
for epoch in range(n_epochs):
    # ... rest of your code ...

    for name, param in model.named_parameters():
        param_np = param.clone().cpu().data.numpy()
        if param_np.size != 0:  # Check if the tensor is not empty
            image = plot_to_tensor()
            writer.add_image(f'Histogram/{name}', image, global_step=epoch)
        else:
            print(f"Skipping histogram for {name} because it's empty.")
