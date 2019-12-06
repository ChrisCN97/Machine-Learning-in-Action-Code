# Machine Learning in Action Code
Rewriting runnable Python3 code for Python2 samples in the book
> [Official Code](https://www.manning.com/books/machine-learning-in-action)
* Start on 2019.11.21
* Finish on 2019.12.5

## 笔记
### 1.Classify（分类）
#### 2-kNN（k-NearestNeighbor）
* 分类算法
* 算得欧式距离最近的k个样本，统计k个样本中占比最大的Label，以此Label作为分类结果
* 优点：对异常值不敏感
* 缺点：复杂度高，模型无法保存
#### 3-tree（决策树）
* ID3：将无序的数据变得更有序。使用香农熵（entropy）来计算数据的混乱程度，熵越大，数据越混乱。每次迭代中，找出最有分类能力的特征，也就是分类后，熵下降最多，信息增益越大的特征，以此特征建立树节点。直到每个节点下都是相同的分类值为止。
* 优点：结构易理解
* 缺点：可能过渡匹配，ID3无法直接处理数值型数据
#### 4-naiveBayes（朴素贝叶斯）
* 朴素：假设各随机变量相互独立，所以 p(w1, w2, ..., wn | ci) 可拆分为 p(w1 | ci) * p(w2 | ci) * ...* p(wn | ci)
* 计算各随机变量的概率，通过贝叶斯准则进行计算
* 优点：数据较少时也有效
* 缺点：对输入数据的准备方式比较敏感
### 5-logisticRegression（逻辑回归）
* 二值型输出分类器
* 利用sigmoid函数，进行梯度下降/上升，还可以使用随机梯度下降/上升进行求解
### 6-svm（支持向量机）
* 一般使用软件包
* **支持向量**就是离**分隔超平面**最近的那些点，svm的任务就是最大化支持向量到分割面的距离
* 求解涉及到[拉格朗日乘子法和KKT条件](https://blog.csdn.net/on2way/article/details/47729419)
* 核函数可选用高斯函数，通过核函数解决非线性可分的情况
* 优点：可泛化，流行
* 缺点：对参数调节和核函数敏感
### 7-AdaBoost（一种元算法）
* 元算法有两种：bagging（打乱数据集给分类器进行权重相等的投票，随机森林），boosting（针对上一分类器错分的数据进行分类，加权求结果）
* AdaBoost是adaptive boosting，利用分类错误率来调整样本的权重，组合简单的单层决策树（Decision Stump）进行分类
* 优点：针对错误的调节能力强
### 非均衡问题
* 使用混淆矩阵（confusion matrix），分类TP（正确，预测真，样本真），FP（错误，预测真，样本假），FN， TN
* precision（查准率）：TP/(TP+FP)
* recall（查全率）：TP/(TP+FN)
## 2.Regression（回归）
### 8-linearRegression（线性回归）
* 利用正规方程求解（满足：线性回归，n < 10000，可逆（非奇异，X.T * X 中，m>n））
* 可以对数据进行加权，越接近数据点的样本权重越高，从而获得更好的预测值
* 可以使用岭回归（+λI）使矩阵非奇异，进行参数缩减（shrinkage），去掉不重要的参数
* 使用corrcoef()计算预测值和样本值的[相关系数](https://blog.csdn.net/zzh1301051836/article/details/82217676)
* 使用逐步线性回归，观察参数变化趋势，从而及时找出最重要的参数
### 9-treeRegression（树回归）
* 使用CART（Classification And Regression Trees），使用二元切分来处理连续变量，利用总方差来计算数据的混乱程度
* 使用均值或是线性回归来构建叶子节点
* 预剪枝：当误差下降很少或是节点样本过少时，停止切分
* 后剪枝：如果合并叶子节点会降低误差，就合并
## 3.无监督学习
### 10-kMean（k-均值聚类）
* 随机选k个簇心，每次迭代为每条数据找到最近的簇心，然后根据簇均值重新计算簇心位置，知道簇心不再变动
* 二分k-均值：每次循环，对每个簇进行二分聚类，然后选出误差变化最大的簇进行二分聚类，直到簇数量足够
### 11-Apriori
* 支持度（support）：数据集中包含改项集的记录所占的比例
* 置信度（confidence）：confidence（A->B） = support（A，B）/support（A）
* Apriori用来发掘频繁项集，原理：一个项集是非频繁集，那么它的所有超集也是非频繁的。
* 第k次循环，挖掘出长度为k的项集Ck，扫描整个数据集，得到长度为k的频繁项集Lk
* 挖掘关联规则：根据置信度公式对频繁项集进行遍历，满足：某规则（012->3）不满足最小可信度要求，则其所有子集（01->23，02->13）也不满足
* 缺点：数据集要遍历多次
### 12-fpGrowth（Frequent Pattern）
* 优点：构建FP数寻找频繁项集，只要遍历两次数据集
* 首先排除出现次数过少的元素，然后构建FP树，记录每条排序好的数据的出现次数
* 对每个出现在FP树中的元素x，用x的前序路径构建FP树，每次都能过滤掉出现次数过少的路径，迭代这一过程，最终得到所有频繁项
## 4.数据压缩
### 13-PCA
* principal component analysis，利用协方差的特征矩阵进行数据降维
### 14.0-推荐系统
* 已知user已评价过的物品集合x，预测user未评价的物品y。利用所有人对x的评价和对y的评价，计算x和y的相似度作为权重，累加x的评分。xy越相近，他们的得分就应该越相同
### 14-SVD
* singular value decomposition，奇异值分解，Data(m * n) = U(m * m) * Sigma(m * n) * V.T(n * n)
* Data(m * n) ~= U(m * 3) * Sigma(3 * 3) * V.T(3 * n)，Sigma保留越多，重构越精确，一般保留90%的精确度


## 下一步
* matplotlib
* tkinter
