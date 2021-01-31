import numpy as np
# from sklearn.preprocessing import normalize

# a = [[1,2,3],[3,6,9]]
# b = [[1,2,3],[3,6,9]]
# a = np.array(a)
# # for i in range(a.shape[1]):
# #       c_max = np.max(a[:,i])
# #       c_min = np.min(a[:,i])
# #       a[:,i] = (a[:,i] - c_min) / (c_max - c_min)
# a = [a[:,1:2] for _ in range(10)]
# a = np.hstack(a)

# a = a[0:1,:]

# a = np.where(a<5, a, 5)

# a = normalize(a, axis=0)
# a = a.take([0, 2], 1)
# a = [0, 'a']
# print(a)

# import re
# pattern = 'AUC: (\d+.\d+)'

# a = re.findall(pattern, 'AUC: 0.3252')
# print(a)

a = [1,2,3]
b = [1,2,3]

a = np.array(a)
b = np.array(b)

a = np.hstack((a,b))

print(a)
print(a.shape)

sel = set()
sel = sel.union(set(a))
print(sel)

X = None
if X:
      print(1)
X = np.array(a)
if X is None:
      print(2)