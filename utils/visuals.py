import matplotlib.pyplot as plt


def show_result(images, labels):
    images = images.cpu()
    labels = labels.cpu()
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(18, 7))
    k = 0
    for i, ax in enumerate(axes.ravel()):
        if i % 2 == 0:
            ax.imshow(images[k])
        else:
            ax.imshow(labels[k])
            k += 1
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
