import numpy as np

print("数据分析结果:")
print("-" * 30)

# 训练数据
train_data = np.load('data/train_data.npy')
print("训练数据形状:", train_data.shape)
print("训练数据类型:", train_data.dtype)
print("训练数据范围:", np.min(train_data), "至", np.max(train_data))
print("训练数据样本(第一个样本前5个值):", train_data[0][:5])
print("-" * 30)

# 训练标签
train_labels = np.load('data/train_labels.npy')
print("训练标签形状:", train_labels.shape)
print("训练标签类型:", train_labels.dtype)
print("训练标签唯一值:", np.unique(train_labels))
print("训练标签样本(前10个):", train_labels[:10].flatten())
print("-" * 30)

# 测试数据
test_data = np.load('data/test_data.npy')
print("测试数据形状:", test_data.shape)
print("测试数据类型:", test_data.dtype)
print("测试数据范围:", np.min(test_data), "至", np.max(test_data))
print("-" * 30)

# 测试标签
test_labels = np.load('data/test_labels.npy')
print("测试标签形状:", test_labels.shape)
print("测试标签类型:", test_labels.dtype)
print("测试标签唯一值:", np.unique(test_labels))
print("各类别样本数量:")
for i in range(10):
    print(f"  类别 {i}: {np.sum(train_labels == i)} 训练样本, {np.sum(test_labels == i)} 测试样本") 