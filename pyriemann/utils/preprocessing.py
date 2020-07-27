import numpy as np
import scipy as sp


def make_sliding_subarrays(array, window_size, step_size):
    length = array.shape[0]
    steps = int(np.ceil( min(window_size, length-window_size+1) / step_size))
    result = [[] for i in range(0, steps)]
    for i in range(0, steps):
        split = np.split(array, range(i * step_size, length, window_size), axis=0)
        if len(split) > 1:
            if split[0].shape[0] < window_size:
                split = split[1:]
            if split[-1].shape[0] < window_size:
                split = split[:-1]
            result[i] = np.array(split).swapaxes(1, 2)

        else:
            print(split)
            print(array)
    return np.vstack(result)


def make_sliding_windows(emg, grasp, grasprepetition, intlength=400, step_size=100):
    valid_reps = grasprepetition >= 0
    grasps = np.unique(grasp)
    graspreps = np.unique(grasprepetition)

    n_grasps = grasps.size
    n_reps = graspreps.size
    X = [[] for i in range(n_grasps)]
    y = [[] for i in range(n_grasps)]

    for i, g in enumerate(grasps):
        for k, gr in enumerate(graspreps):
            true_array = (valid_reps * (grasp == g) * (grasprepetition == gr))

            data = emg[true_array, :]
            if data.shape[0] < 400:
                continue
            else:
                X[i].append(make_sliding_subarrays(data, intlength, step_size))
                y[i].append(np.ones(X[i][-1].shape[0]) * g)
        X[i] = np.concatenate(X[i])
        y[i] = np.concatenate(y[i])
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


X = [range(0,100000) for i in range(10)]
y = np.tile(np.repeat(range(10),1000), 10)
print(y)