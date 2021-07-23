import numpy as np
from numpy import linalg

a = np.array([1,2,3,4,5])

b = np.array([-1,-2,-3,-4,-5])

num = a.dot(b)
denom = linalg.norm(a) * linalg.norm(b)
weight_cosine_similarity = num / denom
print('a:b cos sim: {}'.format(weight_cosine_similarity))

num = b.dot(a)
denom = linalg.norm(b) * linalg.norm(a)
weight_cosine_similarity = num / denom
print('b:a cos sim: {}'.format(weight_cosine_similarity))

vm_migration_idxs = [-1] * 5
vm_migration_idxs_array = [vm_migration_idxs] * 5

print(vm_migration_idxs)
print(vm_migration_idxs_array)
print(vm_migration_idxs_array[0])
print(vm_migration_idxs_array[4][0])

a = np.array([1,2,3,4,5,6,7,8])

b = np.array([-1,-2,-3,-4,-5])

a[0:len(b)] = b

print(a)


cur_idxs = np.array([1,3,5])
res_idxs = np.array([1,5,7,9])

idxs_intersection = np.array([idx for idx in cur_idxs if idx in res_idxs])

idxs_complement = np.array([idx for idx in res_idxs if idx not in cur_idxs])

print('cur_idxs: {} res_idxs: {} intersection: {}, complement: {}'.format(cur_idxs, res_idxs, idxs_intersection, idxs_complement))