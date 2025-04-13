import os
import numpy as np
import time
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from modules.model import MultiLayerNetwork, create_mlp
from modules.layers import Dense, BatchNorm, Dropout
from modules.loss import get_loss
from modules.optimizers import get_optimizer
from modules.activations import get_activation
from utils.data_loader import load_data, train_val_split
from utils.preprocessing import normalize_data, one_hot_encode
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_class_accuracy,
    plot_precision_recall_f1
)


def run_best_model_experiment(args):
    """
    Run best model experiment with optimal hyperparameters.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("\n" + "="*50)
    print("运行最佳模型实验（使用最优超参数）")
    print("="*50)
    
    # 创建结果目录
    results_dir = os.path.join('results', 'best_model')
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置随机种子以确保可重现性
    np.random.seed(args.seed)
    
    # 加载最佳超参数
    try:
        with open(os.path.join('results', 'hyperparameter_tuning', 'best_parameters.json'), 'r') as f:
            best_params = json.load(f)
        
        print("\n最佳超参数配置:")
        for param_name, param_value in best_params.items():
            print(f"{param_name}: {param_value}")
    except FileNotFoundError:
        print("\n找不到最佳参数文件，将使用命令行参数...")
        best_params = {
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'n_neurons': args.n_neurons,
            'n_layers': args.n_layers,
            'batch_size': args.batch_size,
            'activation': args.activation,
            'momentum': args.momentum,
            'dropout_rate': args.dropout_rate
        }
    
    # 加载数据
    print("\n加载数据...")
    train_data, train_labels, test_data, test_labels = load_data(args.data_dir)
    
    # 预处理数据
    print("预处理数据...")
    # 标准化特征
    train_data_norm, mean, std = normalize_data(train_data)
    test_data_norm = (test_data - mean) / std
    
    # 将标签转换为one-hot编码
    train_labels_one_hot = one_hot_encode(train_labels, args.num_classes)
    test_labels_one_hot = one_hot_encode(test_labels, args.num_classes)
    
    # 将训练数据分为训练集和验证集
    train_data_subset, train_labels_subset, val_data, val_labels = train_val_split(
        train_data_norm, train_labels_one_hot, args.val_ratio, args.seed)
    
    # 创建模型
    print("\n创建最优模型...")
    model = create_mlp(
        input_dim=train_data.shape[1],
        hidden_dims=[best_params['n_neurons']] * best_params['n_layers'],
        output_dim=args.num_classes,
        activation_name=best_params['activation'],
        use_batch_norm=args.use_batch_norm,
        use_dropout=True,
        dropout_rate=best_params['dropout_rate'],
        weight_scale=args.weight_scale
    )
    
    # 编译模型
    print("编译模型...")
    loss = get_loss(args.loss)
    
    optimizer_args = {
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay']
    }
    
    # 只有在使用SGD动量时才添加动量参数
    if args.optimizer.lower() == 'sgd_momentum':
        optimizer_args['momentum'] = best_params['momentum']
    
    optimizer = get_optimizer(args.optimizer, **optimizer_args)
    model.compile(loss=loss, optimizer=optimizer)
    
    # 打印模型详细信息
    print(f"模型参数总数: {model.count_parameters()}")
    
    # 训练模型
    print("\n训练最佳模型...")
    start_time = time.time()
    
    history = model.train(
        x_train=train_data_subset,
        y_train=train_labels_subset,
        x_val=val_data,
        y_val=val_labels,
        epochs=args.epochs,
        batch_size=best_params['batch_size'],
        shuffle=True,
        use_early_stopping=args.use_early_stopping,
        patience=args.patience,
        use_gradient_clipping=args.use_gradient_clipping,
        clip_value=args.clip_value,
        save_best=True
    )
    
    training_time = time.time() - start_time
    
    # 在测试集上评估模型
    print("\n在测试集上评估最佳模型...")
    metrics = model.evaluate(test_data_norm, test_labels_one_hot)
    
    # 计算收敛速度（达到最终性能90%所需的轮次）
    final_val_acc = history['val_acc'][-1]
    threshold = 0.9 * final_val_acc
    convergence_epoch = next((i for i, acc in enumerate(history['val_acc']) if acc >= threshold), len(history['val_acc']))
    
    # 计算过拟合（训练精度和验证精度之间的差异）
    overfitting = abs(history['train_acc'][-1] - history['val_acc'][-1])
    
    # 打印结果
    print("\n最终结果:")
    print(f"测试准确率: {metrics['accuracy']:.4f}")
    print(f"测试损失: {metrics['loss']:.4f}")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"收敛速度: {convergence_epoch}轮")
    print(f"过拟合度: {overfitting:.4f}")
    print(f"精确率: {metrics['avg_precision']:.4f}")
    print(f"召回率: {metrics['avg_recall']:.4f}")
    print(f"F1分数: {metrics['avg_f1_score']:.4f}")
    
    # 保存模型和结果
    model_dir = os.path.join(results_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存训练历史图表
    plot_training_history(
        history,
        save_path=os.path.join(results_dir, 'training_history.png')
    )
    
    # 获取预测和真实标签
    y_pred = model.predict(test_data_norm)
    
    # 处理输出的维度问题
    if len(y_pred.shape) == 1:
        # 如果是一维数组，可能是二分类问题的输出
        if args.num_classes == 2:
            # 二分类问题，将输出转换为类别
            y_pred_classes = (y_pred > 0.5).astype(int)
        else:
            # 如果不是二分类问题但输出是一维的，可能是已经是类别索引
            y_pred_classes = y_pred.astype(int)
    else:
        # 正常的多分类输出，取最大值的索引
        y_pred_classes = np.argmax(y_pred, axis=1)
    
    y_true = np.argmax(test_labels_one_hot, axis=1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # 计算每个类别的精确率、召回率和F1分数
    class_metrics = {
        'confusion_matrix': cm,
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for i in range(args.num_classes):
        # 精确率：真正例/(真正例+假正例)
        precision_i = cm[i, i] / max(np.sum(cm[:, i]), 1)
        # 召回率：真正例/(真正例+假负例)
        recall_i = cm[i, i] / max(np.sum(cm[i, :]), 1)
        # F1分数：2 * 精确率 * 召回率 / (精确率 + 召回率)
        f1_i = 2 * precision_i * recall_i / max(precision_i + recall_i, 1e-8)
        
        class_metrics['precision'].append(precision_i)
        class_metrics['recall'].append(recall_i)
        class_metrics['f1_score'].append(f1_i)
    
    try:
        # 绘制混淆矩阵
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred_classes,
            num_classes=args.num_classes,
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
        
        # 绘制每个类别的准确率
        plot_class_accuracy(
            metrics=class_metrics,
            num_classes=args.num_classes,
            save_path=os.path.join(results_dir, 'class_accuracy.png')
        )
        
        # 绘制精确率、召回率和F1分数
        plot_precision_recall_f1(
            metrics=class_metrics,
            num_classes=args.num_classes,
            save_path=os.path.join(results_dir, 'precision_recall_f1.png')
        )
    except Exception as e:
        print(f"绘图时出错: {e}")
    
    # 保存结果
    results = {
        'hyperparameters': best_params,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'loss': float(metrics['loss']),
            'precision': float(metrics['avg_precision']),
            'recall': float(metrics['avg_recall']),
            'f1_score': float(metrics['avg_f1_score']),
            'training_time': float(training_time),
            'convergence_speed': int(convergence_epoch),
            'overfitting': float(overfitting),
            'num_parameters': int(model.count_parameters())
        }
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n最佳模型实验完成！")
    return results


def run_advanced_best_model_experiment(args):
    """
    Run advanced best model experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("Advanced best model experiment not implemented yet.")
    pass
