import matplotlib.pyplot as plt


def show_result(images, output, labels):
    images = images.cpu()
    outputs = output.cpu()
    labels = labels.cpu()
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(18, 7))
    k = 0
    for i, ax in enumerate(axes.ravel()):
        if i % 3 == 0:
            ax.imshow(images[k].detach().permute(1, 2, 0).numpy())
        elif i % 3 == 1:
            ax.imshow(labels[k].detach().permute(1, 2, 0).numpy())
        else:
            ax.imshow(outputs[k].detach().permute(1, 2, 0).numpy())
            k += 1
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
