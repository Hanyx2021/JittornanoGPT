import re
import matplotlib.pyplot as plt

def parse_lr_from_log(file_path):
    """解析日志文件中的学习率值"""
    iter_nums = []
    lr_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'iter = (\d+), lr = \[([\d.]+), [\d.]+\]', line)
            if match:
                iter_nums.append(int(match.group(1)))
                lr_values.append(float(match.group(2)))
    return iter_nums, lr_values

def plot_lr_curves(files):
    """绘制学习率曲线"""
    plt.figure(figsize=(10, 6))
    for file_path in files:
        iter_nums, lr_values = parse_lr_from_log(file_path)
        plt.plot(iter_nums, lr_values, label=file_path.split('.')[0])
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curve')
    plt.legend()
    plt.savefig('learning_rate.png')

plot_lr_curves(['nanoGPT-Jittor.log', 'nanoGPT-Pytorch.log'])
