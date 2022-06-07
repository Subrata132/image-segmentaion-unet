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


def show(img, output, label, denorm=False):
    img, output, label = img.cpu(), output.cpu(), label.cpu()
    fig, ax = plt.subplots(len(output), 3, figsize=(10, 10))

    for i in range(len(output)):
        if len(output) == 3:
            img, lab, act = img[i], output[i], label[i]
            img, lab, act = img, lab.detach().permute(1, 2, 0).numpy(), act
            ax[i][0].imshow(img.permute(1, 2, 0))
            ax[i][1].imshow(lab)
            ax[i][2].imshow(act.permute(1, 2, 0))
        else:
            img, lab, act = img[i], output[i], label[i]
            img, lab, act = img, lab.detach().permute(1, 2, 0).numpy(), act
            ax[0].imshow(img.permute(1, 2, 0))
            ax[1].imshow(lab)
            ax[2].imshow(act.permute(1, 2, 0))
    plt.show()
