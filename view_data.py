import numpy as np

# 查看训练数据
train_data = np.load('data/train_data.npy')
print("训练数据形状:", train_data.shape)
print("训练数据类型:", train_data.dtype)
print("训练数据样本(第一个):", train_data[0])

# 查看训练标签
train_labels = np.load('data/train_labels.npy')
print("\n训练标签形状:", train_labels.shape)
print("训练标签类型:", train_labels.dtype)
print("训练标签样本(前10个):", train_labels[:10])
print("训练标签唯一值:", np.unique(train_labels))

# 查看测试数据
test_data = np.load('data/test_data.npy')
print("\n测试数据形状:", test_data.shape)
print("测试数据类型:", test_data.dtype)

# 查看测试标签
test_labels = np.load('data/test_labels.npy')
print("\n测试标签形状:", test_labels.shape)
print("测试标签类型:", test_labels.dtype)
print("测试标签唯一值:", np.unique(test_labels)) 