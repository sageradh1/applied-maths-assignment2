import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)

    # Implementation 1
    mse = np.mean((pred - tar)**2)
    print("mse1:\n", mse)

    # Implementation 2
    # mse = np.square(np.subtract(targets,predictions)).mean()
    # print("mse2:\n", mse)
    rmse = np.sqrt(mse)
    
    return rmse