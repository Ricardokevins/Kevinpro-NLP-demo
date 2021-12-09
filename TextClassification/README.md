# TextClassification

文本分类demo，还同时作为baseline对比，加上了很多NLP trick实践

## 模型

1. bert-base-chinese
2. Base RNN
3. Base RNN + Attention
4. Base transformer
5. CHI + TFIDF + LR/MLP/SVM (相对传统的统计学习方法)
   
## 技术

1. 对抗训练
2. 模型蒸馏
3. 预训练
4. Pytorch_lighting框架，混合精度训练(LightingMain.py)

# 神经网络算法使用说明

这个文件夹下主要是句子分类的代码

同时我集成了对抗学习和知识蒸馏的功能代码

由于**我比较懒，所以没有用命令行来实现相关的功能选择**

即需要直接通过修改代码来实现不同的功能，具体操作如下



## 模型选择

在main文件中直接修改代码即可，相关代码如下

```python
# Choose your model here
net = BiRNN()
# net = BertClassifier()
# net = BiLSTM_Attention1()
# net = BiLSTM_Attention2()
# net = TransformerClasssifier()
net = net.cuda()
```



## 功能选择和使用

代码中有三个函数，通过修改main里的注释情况可以选择

```
train ----> 基本的基准训练函数
train_kd ----> 知识蒸馏训练函数
train_with_FGM ----> 对抗训练函数
```



## 超参数选择

都硬编码在代码里了，包括学习率等等的，可以自己稍作阅读修改变量即可

还有一些杂七杂八的，比如模型保存位置啥的，都可以自己修改和自定义



##  其他

基准训练和对抗训练都是直接修改文件就可以使用了

知识蒸馏我使用了微调后的bert作为teacher model，所以需要自己准备后放在目录里

# 统计机器学习算法
先用CHI算法做特征筛选，然后用BOW,OneHot,TFIDF作为特征填充

最后用LR,MLP,SVM等传统机器学习算法进行分类，效果很好

实现于CHIClassifier.py