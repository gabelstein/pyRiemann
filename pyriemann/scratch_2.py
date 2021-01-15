
from utils.mean import mean_covariance, mean_logdet
#from utils.tangentspace import tangent_space, transport
#from utils.covariance import covariances, covariances_par
#from utils.distance import distance_riemann, distance_riemann_par
#from utils.geodesic import geodesic
import numpy as np
from sklearn.datasets import make_spd_matrix
from numba import prange, njit
import time

N = 10
covmats = np.array([make_spd_matrix(8) for i in range(N)])
mean = make_spd_matrix(8)

#@njit
def calc_par(covmat,mean):
    mean_logdet(covmat)

"""
start = time.time()
cov = calc(covmats,mean)
end = time.time()
print(f"timeseq: {(end-start):.2f}")

start = time.time()
cov = calc(covmats,mean)
end = time.time()
print(f"timeseq: {(end-start):.2f}")"""

for i in range(5):
    start = time.time()
    cov = calc_par(covmats, "logdet")
    end = time.time()
    print(f"timepar: {(end - start):.2f}")



"""


covmats = np.array([make_spd_matrix(8) for i in range(500000)])


@njit
def calc_par(covmat):
    for i in prange(500000-1):
        distance_riemann_par(covmat[i],covmat[i+1])


def calc(covmat):
    for i in range(500000-1):
        distance_riemann(covmat[i],covmat[i+1])

start = time.time()
cov = calc(covmats)
end = time.time()
print(f"timeseq: {(end-start):.2f}")

start = time.time()
cov = calc_par(covmats)
end = time.time()
print(f"timepar: {(end - start):.2f}")

start = time.time()
cov = calc_par(covmats)
end = time.time()
print(f"timepar: {(end - start):.2f}")


raw_data = np.random.rand(1000000,4)
step = 10
intlength = 200
datachunk = raw_data
windows = np.array([raw_data[i*step:i*step+intlength].T
                         for i in range(datachunk.shape[0]//step-intlength//step)]
                   ).reshape((-1, datachunk.shape[1], intlength))

start = time.time()
cov = covariances(windows, "oas")
end = time.time()
print(f"timeseq: {(end-start):.2f}")

start = time.time()
cov_par = covariances_par(windows)
end = time.time()
print(f"timepar: {(end-start):.2f}")
print(np.sum(cov_par-cov))

start = time.time()
cov_par = covariances_par(windows)
end = time.time()
print(f"timepar: {(end-start):.2f}")
print(np.sum(cov_par-cov))




covmats = np.array([make_spd_matrix(8) for i in range(100000)])
mean_r = make_spd_matrix(8)



start = time.time()
ts = tangent_space(covmats,mean_r)
end = time.time()
print(f"timeseq: {(end-start):.2f}")

start = time.time()
ts_par = tangent_space_par(covmats,mean_r)
end = time.time()
print(f"timepar: {(end-start):.2f}")

for i in range(3):
    start = time.time()
    ts_par = tangent_space_par(covmats,mean_r)
    end = time.time()
    print(f"timepar: {(end-start):.2f}")

print(np.sum(ts-ts_par))

"""