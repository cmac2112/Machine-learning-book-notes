import matplotlib.pyplot as plt
class ImageHelpers:
    def __init__(self, data):
        self.data = data

    def plot_digit(self):
        image = self.data.reshape(28,28)
        plt.imshow(image, cmap='binary')
        plt.axis('off')

    def plot_array(self):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()

        for i in range(10):
            image = self.data[i].reshape(28, 28)
            axes[i].imshow(image, cmap='binary')
            axes[i].axis('off')
            plt.savefig(f'digit_{i}.png')

        plt.close()