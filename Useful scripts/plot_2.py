def plot_2(x,y,z,label_y,label_z,title):
    import matplotlib.pyplot as plt
    plt.plot(x,y,'b-',label=label_y)
    plt.plot(x,z,'r--',label=label_z)
    plt.title(title)
    plt.legend()
