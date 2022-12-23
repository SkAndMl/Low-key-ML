import numpy as np

def window_data(data, window=7, horizon=1):
    """
    Function to window data
    
    Parameters
    -------------
    window - dimensionality of the timesteps
    horizon - number of instances to predict using 
              past window size timesteps

    Returns: X - np.array of shape [None, window]
             y - np.array of shape [None, horizon]
    """
    if len(data)-window-horizon+1 < 0:
        raise ValueError("Either the window or the horizon is too large \
                          for the given data")
   
   X = np.empty(shape=(len(data)-window-horizon+1,window))
    y = np.empty(shape=(len(data)-window-horizon+1, horizon))
    

    for i in range(0,len(X),1):
        X[i,:] = data[i:i+window]
        y[i,:] = data[i+window:i+window+horizon]

    return X,y


if __name__ == '__main__':
    data = np.arange(1,21)
    window = int(input("Enter the window size: "))
    horizon = int(input("Enter the horizon size: "))
    X,y = window_data(data, window=window, horizon=horizon)
    
    print(f"X: {X}\ny: {y}")

