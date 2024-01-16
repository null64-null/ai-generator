# 1. for mnist (94%)
'''
layers = [
    AffineLayer([784, 50]),
    Sigmoid(),
    AffineLayer([50, 10]),
    Softmax(),
]
'''

# 2. best for mnist (98%)
'''
# 99%
layers = [
    AffineLayer([784, 500]),
    Relu(),
    AffineLayer([500, 100]),
    Relu(),
    AffineLayer([100, 10]),
    Softmax(),
]
'''

