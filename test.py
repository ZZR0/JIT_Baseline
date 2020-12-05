import numpy as np
from sklearn.preprocessing import normalize

a = [[1,2],[
      3,6]]
a = np.array(a)
a = normalize(a, axis=0)

print(a)