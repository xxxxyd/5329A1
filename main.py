import argparse
import sys

from experiments.hyperparameter_tuning import (
    run_learning_rate_experiment,
    run_weight_decay_experiment,
    run_neurons_experiment,
    run_layers_experiment,
    run_batch_size_experiment,
    run_activation_experiment,
    run_momentum_experiment,
    run_dropout_experiment,
    run_all_hyperparameter_experiments
)

from experiments.best_model import run_best_model_experiment


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='神经网络超参数调优与最佳模型训练')
    
    # 添加通用实验参数
    parser.add_argument('--experiment', type=str, default='best_model',
                        choices=['learning_rate', 'weight_decay', 'neurons', 'layers', 'batch_size', 
                                 'activation', 'momentum', 'dropout_rate', 'all', 'best_model'],
                        help='要运行的实验')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录路径')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='分类类别数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    
    # 模型参数
    parser.add_argument('--n_layers', type=int, default=2,
                        help='隐藏层数量')
    parser.add_argument('--n_neurons', type=int, default=128,
                        help='每层神经元数量')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu', 'gelu'],
                        help='激活函数')
    parser.add_argument('--weight_scale', type=float, default=0.01,
                        help='权重初始化比例')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='是否使用批量归一化')
    parser.add_argument('--use_dropout', action='store_true',
                        help='是否使用dropout')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout比率')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='sgd_momentum',
                        choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop'],
                        help='优化器类型')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='权重衰减系数')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量系数')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='损失函数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮次')
    parser.add_argument('--use_early_stopping', action='store_true',
                        help='是否使用早停')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--use_gradient_clipping', action='store_true',
                        help='是否使用梯度裁剪')
    parser.add_argument('--clip_value', type=float, default=5.0,
                        help='梯度裁剪阈值')
    
    args = parser.parse_args()
    
    # 根据选择的实验运行相应的函数
    if args.experiment == 'learning_rate':
        run_learning_rate_experiment(args)
    elif args.experiment == 'weight_decay':
        run_weight_decay_experiment(args)
    elif args.experiment == 'neurons':
        run_neurons_experiment(args)
    elif args.experiment == 'layers':
        run_layers_experiment(args)
    elif args.experiment == 'batch_size':
        run_batch_size_experiment(args)
    elif args.experiment == 'activation':
        run_activation_experiment(args)
    elif args.experiment == 'momentum':
        run_momentum_experiment(args)
    elif args.experiment == 'dropout_rate':
        run_dropout_experiment(args)
    elif args.experiment == 'all':
        run_all_hyperparameter_experiments(args)
    elif args.experiment == 'best_model':
        run_best_model_experiment(args)
    else:
        print(f"未知实验: {args.experiment}")
        sys.exit(1)
    
    print("\n实验完成!")


if __name__ == '__main__':
    main()
