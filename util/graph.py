import matplotlib.pyplot as plt

def chart(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0)
    plt.ylim(0)
    plt.show()
