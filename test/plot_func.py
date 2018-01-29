import matplotlib.pyplot as plt
import matrixGeneration

def plot_func(vector):
    """
    plt.plot(vector)
    plt.ylabel('something')
    plt.xlabel('something else')
    plt.title('helloo')
    plt.show()
    """
    print(vector)

array = matrixGeneration.random_array()

plot_func(array)