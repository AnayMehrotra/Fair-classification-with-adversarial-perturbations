import random
from copy import deepcopy
random.seed()

# insert flipping noises in the selected feature
def flipping(feature_names, features, name, eta0, eta1):
    index = feature_names.index(name)
    N = features.shape[0]
    noisyfea = deepcopy(features[:,index])
    count = 0
    for i in range(N):
        seed = random.random()
        if int(features[i][index]) == 1:
            if seed < eta1:
                noisyfea[i] = 1 - noisyfea[i]
                count += 1
        elif int(features[i][index]) == 0:
            if seed < eta0:
                noisyfea[i] = 1 - noisyfea[i]
                count += 1
    print('Count_flipping:', count)
    return index, noisyfea