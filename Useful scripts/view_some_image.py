def view_some_image(X,y,r_size=28,cmap="gray"):
  """
  This function displays random images from the passed features.
  X -> features or pixel values.
  y -> target value
  size -> it is the dimension of the image
  cmap -> specifies color map of the image
  """
  import numpy as np
  import  matplotlib.pyplot as plt
  random_index = np.random.randint(0,len(X))
  rand_img = X[random_index].reshape(r_size,r_size)
  plt.title(f"Number: {y[random_index]}",size=15)
  plt.imshow(rand_img,cmap=cmap)
  plt.axis("off")