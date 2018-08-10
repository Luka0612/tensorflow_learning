# coding: utf-8

"""
hyperopt是采用贝叶斯优化基于模型的用于寻找函数最小值的方法

https://www.jiqizhixin.com/articles/2018-08-08-2
https://github.com/FontTian/hyperopt-doc-zh/wiki/%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3%E5%AF%BC%E8%AF%BB
"""
import hyperopt


# define an objective function
def objective(args):
    case, val = args["args"]
    if case == 'case 1':
        return val
    else:
        return val ** 2


# define a search space
from hyperopt import hp

args = hp.choice('a',
                  [
                      ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                      ('case 2', hp.uniform('c2', -10, 10))
                  ])
space = {"args": args, "class_weight": hp.choice('class_weight', [None, 'balanced'])}

# minimize the objective over the space
from hyperopt import fmin, tpe

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print best
# -> {'a': 1, 'c2': 0.01420615366247227}
print hyperopt.space_eval(space, best)
# -> {'case 2', 0.01420615366247227}
