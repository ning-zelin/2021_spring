"""
对于分类问题, 如果各类的数据量差距较大, 对损失函数贡献不同,
会导致模型更喜欢预测出数据量大的类, 进而影响精度.
解决方法: 给cross entropy加一个权重
	 criterion = nn.CrossEntropyLoss(weight=config['weights'])
获取weights的方法如下:
"""

"""
细节:	前面加一个get_device函数或者改一下此函数
method == 1: 比较好的经验法则
"""
def get_weights(train_y, method=1):
    _cnt = Counter(train_y).items()
    _cnt = [(int(e[0]), e[1]) for e in _cnt]
    _cnt = sorted(_cnt, key=itemgetter(0))
    nSamples = [e[1] for e in _cnt]
    total = sum(nSamples)
#    print(total)
#    print(nSamples)
    
    device = get_device()
    
	if method == 0:
        weights = None
    elif method == 1:
        weights = [1 - (x / total) for x in nSamples]
        print(weights)
    elif method == 2:
        percentage = [ x / total for x in nSamples]
        print(percentage)
        weights = []
        for p in percentage:
            if p > 0.1:
                weights.append(0.75)
            elif p < 0.01:
                weights.append(1.25)
            else:
                weights.append(1)
    weights = torch.FloatTensor(weights).to(device)
    return weights

weights = get_weights(train_y)
print(weights)