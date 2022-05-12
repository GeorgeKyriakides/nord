import time
from nord.data_curators.images import get_cifar10


t = time.time()
d = get_cifar10()
print(time.time()-t)
