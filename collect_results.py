import os
import json
import glob

def extract_results(experiment_dir, is_ablation=False):
    results = {}
    
    if is_ablation:
        overall_file = os.path.join('results', 'ablation_study', 'overall_results.json')
    else:
        overall_file = os.path.join('results/hyperparameter_tuning', experiment_dir, 'overall_results.json')
    
    if os.path.exists(overall_file):
        with open(overall_file, 'r') as f:
            results = json.load(f)
    
    return results

def format_results(experiment_name, results, is_ablation=False):
    if is_ablation:
        output = f'\n\n### 消融实验结果\n\n'
        output += '| 配置 | 准确率 | 损失 | 精确率 | 召回率 | F1分数 | 训练时间(秒) | 收敛速度 | 过拟合程度 |\n'
        output += '|--------|--------|------|--------|--------|--------|--------------|----------|------------|\n'
        
        # 按准确率排序
        sorted_keys = sorted(results.keys(), key=lambda x: results[x].get('accuracy', 0), reverse=True)
        
        for key in sorted_keys:
            result = results[key]
            config_name = key
            
            # 格式化结果数据
            accuracy = result.get('accuracy', 'N/A')
            if isinstance(accuracy, (int, float)):
                accuracy = f'{accuracy:.4f}'
            
            loss = result.get('loss', 'N/A')
            if isinstance(loss, (int, float)):
                loss = f'{loss:.4f}'
            
            precision = result.get('precision', 'N/A')
            if isinstance(precision, (int, float)):
                precision = f'{precision:.4f}'
            
            recall = result.get('recall', 'N/A')
            if isinstance(recall, (int, float)):
                recall = f'{recall:.4f}'
            
            f1 = result.get('f1_score', 'N/A')
            if isinstance(f1, (int, float)):
                f1 = f'{f1:.4f}'
            
            training_time = result.get('training_time', 'N/A')
            if isinstance(training_time, (int, float)):
                training_time = f'{training_time:.2f}'
            
            convergence = result.get('convergence_speed', 'N/A')
            if isinstance(convergence, (int, float)):
                convergence = f'{convergence:.2f}'
            
            overfitting = result.get('overfitting', 'N/A')
            if isinstance(overfitting, (int, float)):
                overfitting = f'{overfitting:.4f}'
            
            output += f'| {config_name} | {accuracy} | {loss} | {precision} | {recall} | {f1} | {training_time} | {convergence} | {overfitting} |\n'
    else:
        output = f'\n\n### {experiment_name} 超参数实验结果\n\n'
        output += '| 参数值 | 准确率 | 损失 | 精确率 | 召回率 | F1分数 | 训练时间(秒) | 收敛速度 | 过拟合程度 | 参数数量 |\n'
        output += '|--------|--------|------|--------|--------|--------|--------------|----------|------------|----------|\n'
        
        # 对参数值进行排序以便于阅读
        sorted_results = {}
        param_name = experiment_name.lower().replace(' ', '_')
        
        if param_name == '激活函数':
            # 激活函数按字母顺序排序
            sorted_keys = sorted(results.keys())
        else:
            # 数值参数按数值大小排序
            try:
                sorted_keys = sorted(results.keys(), key=lambda x: float(x.split('_')[-1].replace('_', '.')))
            except:
                sorted_keys = sorted(results.keys())
        
        for key in sorted_keys:
            result = results[key]
            # 提取参数值
            if param_name == '激活函数':
                param_value = key.replace(f'{param_name}_', '')
            else:
                param_value = key.split('_')[-1].replace('_', '.')
            
            # 格式化结果数据
            accuracy = result.get('accuracy', 'N/A')
            if isinstance(accuracy, (int, float)):
                accuracy = f'{accuracy:.4f}'
            
            loss = result.get('loss', 'N/A')
            if isinstance(loss, (int, float)):
                loss = f'{loss:.4f}'
            
            precision = result.get('precision', 'N/A')
            if isinstance(precision, (int, float)):
                precision = f'{precision:.4f}'
            
            recall = result.get('recall', 'N/A')
            if isinstance(recall, (int, float)):
                recall = f'{recall:.4f}'
            
            f1 = result.get('f1_score', 'N/A')
            if isinstance(f1, (int, float)):
                f1 = f'{f1:.4f}'
            
            training_time = result.get('training_time', 'N/A')
            if isinstance(training_time, (int, float)):
                training_time = f'{training_time:.2f}'
            
            convergence = result.get('convergence_speed', 'N/A')
            if isinstance(convergence, (int, float)):
                convergence = f'{convergence:.2f}'
            
            overfitting = result.get('overfitting', 'N/A')
            if isinstance(overfitting, (int, float)):
                overfitting = f'{overfitting:.4f}'
            
            params = result.get('params', 'N/A')
            
            output += f'| {param_value} | {accuracy} | {loss} | {precision} | {recall} | {f1} | {training_time} | {convergence} | {overfitting} | {params} |\n'
    
    return output

# 实验名称映射
experiment_names = {
    'learning_rate': '学习率',
    'neurons': '神经元数量',
    'layers': '隐藏层数量',
    'weight_decay': '权重衰减',
    'batch_size': '批量大小',
    'activation': '激活函数',
    'momentum': '动量参数',
    'dropout_rate': 'Dropout率'
}

# 收集所有实验结果
with open('all_experimental_results.txt', 'w', encoding='utf-8') as f:
    f.write('# 全部实验结果汇总\n')
    
    # 添加超参数调优实验结果
    f.write('\n## 超参数调优实验\n')
    for exp_dir in ['learning_rate', 'neurons', 'layers', 'weight_decay', 'batch_size', 'activation', 'momentum', 'dropout_rate']:
        results = extract_results(exp_dir)
        if results:
            formatted_results = format_results(experiment_names[exp_dir], results)
            f.write(formatted_results)
    
    # 添加消融实验结果
    f.write('\n## 消融实验\n')
    ablation_results = extract_results('', is_ablation=True)
    if ablation_results:
        formatted_ablation_results = format_results('', ablation_results, is_ablation=True)
        f.write(formatted_ablation_results)

print('所有实验结果已保存到 all_experimental_results.txt 文件')
