from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
from cs231n.gradient_check import eval_numerical_gradient_array
import numpy as np
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
np.random.seed(231)
x = np.array(range(100)).reshape(100,-1)
w = np.array([[1]])
b = np.array([-5])
dout = np.random.randn(100,1)

out, cache = affine_relu_forward(x, w, b)
# cache contains x and wx+b
dx, dw, db = affine_relu_backward(out, cache)

dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, out)
dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, out)
db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, out)

# Relative error should be around e-10 or less
print('Testing affine_relu_forward and affine_relu_backward:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))