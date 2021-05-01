import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter

def plot_counter_hist(y):
    plt.figure(figsize=(10, 6))
    val_cnt = Counter(y).items()
    val_cnt = [(int(e[0]), e[1]) for e in val_cnt]
    val_cnt = sorted(val_cnt, key=itemgetter(0))
    val_cnt = Counter(dict(val_cnt))
    plt.bar(val_cnt.keys(), val_cnt.values())
    plt.show()
"""
input size : (n,)

example:
plot_counter_hist(np.random.randint(1, 14, 1009))
plt.show()
"""


"""
    val_cnt = [(int(e[0]), e[1]) for e in val_cnt]
上一行的输出[:, 0]号元素为str类型, 要将其转换为int类型才能排序
"""


"""
	val_cnt = sorted(val_cnt, key=itemgetter(0))

sorted: 排序,
sorted(input, key = itemgetter(索引))

examples:

a = np.array([[3,4],
			  [5,2]])
getitm = itemgetter(1)
getitm(a) = [5, 6]

sorted(a, key = itemgetter(0))
-->array([3,4],[5,2])

sorted(a, key = itemgetter(1))
-->array([5,2],[3,4])

"""