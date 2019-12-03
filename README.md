# Machine Learning in Action Code
Rewriting runnable Python3 code for Python2 samples in the book
> [Official Code](https://www.manning.com/books/machine-learning-in-action)
* Start on 2019.11.21

## 笔记
### 监督学习
#### 2-kNN（k-NearestNeighbor）
* 分类算法
* 算得欧式距离最近的k个样本，统计k个样本中占比最大的Label，以此Label作为分类结果
* 优点：对异常值不敏感
* 缺点：复杂度高，模型无法保存
* 适用范围：标称型
#### 3-tree（决策树）
* ID3：将无序的数据变得更有序。使用香农熵（entropy）来计算数据的混乱程度，熵越大，数据越混乱。每次迭代中，找出最有分类能力的特征，也就是分类后，熵下降最多，信息增益越大的特征，以此特征建立树节点。直到每个节点下都是相同的分类值为止。
* 优点：结构易理解
* 缺点：可能过渡匹配，ID3无法直接处理数值型数据
* 适用范围：标称型
#### 4-naiveBayes（朴素贝叶斯）
* 朴素：假设各随机变量相互独立，所以 p(w1, w2, ..., wn | ci) 可拆分为 p(w1 | ci) * p(w2 | ci) * ...* p(wn | ci)
* 计算各随机变量的概率，通过贝叶斯准则进行计算
* 优点：数据较少时也有效
* 缺点：对输入数据的准备方式比较敏感
* 使用范围：标称型

## 数学原理
* 条件概率
* 贝叶斯准则

## 下一步
* matplotlib
* tkinter
