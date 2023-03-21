import torch
x = torch.tensor([[1], [2], [3]])
print(x.size())
print('x.expand(3, 4):', x.expand(3, 4))
print('x.expand(-1, 4):', x.expand(-1, 4))
print('\n####################################################\n')

import paddle
x = paddle.to_tensor([[1], [2], [3]])
print('x.expand(3, 4):', x.expand(shape=[3, 4]))
print('x.expand(-1, 4):', x.expand(shape=[-1, 4]))

'''
import paddle
data = paddle.to_tensor([1, 2, 3], dtype='int32')
out = paddle.expand(data, shape=[2, 3])
print(out)
'''


'''
import paddle
x = paddle.to_tensor([[1], [2], [3]], dtype='int32')
#x = x.numpy()
#print(x.size())
print('x.expand(3, 4):', x.expand(3, 4))
print('x.expand(-1, 4):', x.expand(-1, 4))
'''


