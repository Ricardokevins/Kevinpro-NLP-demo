# NLP

个人的NLP实践demo。部分来源于其他开源项目。欢迎Star Fork。

主要内容

1. **文本分类**
2. **摘要生成**
3. **对话翻译**
4. **GNN实现使用**
5. **Mask Language Model预训练**
6. **其他的NLP炼丹技巧实践**

其他参考实践

1. **bert关系抽取**：[Ricardokevins/Bert-In-Relation-Extraction: 使用Bert完成实体之间关系抽取 (github.com)](https://github.com/Ricardokevins/Bert-In-Relation-Extraction)
2. **文本语意匹配**：[Ricardokevins/Text_Matching: NLP2020中兴捧月句子相似度匹配 (github.com)](https://github.com/Ricardokevins/Text_Matching)
3. **Transfomer实现和其他部件**：[Ricardokevins/EasyTransformer: Quick start with strong baseline of Bert and Transformer without pretrain (github.com)](https://github.com/Ricardokevins/EasyTransformer)

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



# GNN

使用torch_geometric作为图神经网络的工具

自己实现了读取构造图数据，以及图模型的代码

代码可运行，但是效果不好（正在排查原因中）



# Pretrain

## 说明

使用上面句子分类中的语料做Mask Language Model的预训练，采用和BERT一样的预训练策略，对我的Transformer进行预训练

测试

1. 在句子分类任务中，加载预训练后的Transformer
2. 用预训练做Fill Blank任务，见test.py

## 结果

在训练集1000条里构造100000条训练

测试集300条里构造10000条测试

测试集上正确率37%左右

### 句子分类

| Model      | Acc    |
| ---------- | ------ |
| Base       | 81.60% |
| Pretrained | 82.99% |

### 填空

> i will never visit this restaurant again.
>
> i will [MASK] visit this restaurant [MASK]
>
> [MASK] Blank Answer:  never
>
> [MASK] Blank Answer:  again





# ChatBotEnglish

英文对话机器人

### 模型

1. GRU encoder-decoder





# PGNSum

使用PointerGenerator的摘要生成

### 模型

1. LSTM
2. Pointer Generator



# Seq2Seq

普通的Seq2seq翻译网络





# 更新记录

## 2021.1.23

 1. 初次commit 添加句子分类模块，包含Transformer和BiLSTM以及BiLSTM+Attn模型 
 2. 上传基本数据集，句子二分类作为Demo例子
 3. 加上和使用对抗学习思路

## 2021.5.1

1. 重新整理和更新了很多东西.... 略



## 2021.6.22

1. 修复了Text Classification的一些整理问题
2. 增加了Text Classification对应的使用说明



## 2021.7.2

1. 增加了MLM预训练技术实践
2. 修复了句子分类模型里，过分大且不必要的Word Embed（因为太懒，所以只修改了Transformer的）
3. 在句子分类里增加了加载预训练的可选项
4. 修复了一些BUG



# 2021.7.11

1. 增加了GNN在NLP中的应用
2. 实现了GNN在文本分类上的使用
3. 效果不好，暂时怀疑是数据处理的问题

# 2021.7.29

1. 增加了CHI+TFIDF传统机器学习算法在文本分类上的应用
2. 实现和测试了算法性能
3. 更新了README