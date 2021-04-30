# NLP

个人的NLP实践demo。部分来源于其他开源项目。欢迎Star Fork。

主要内容

1. 文本分类
2. 摘要生成
3. 对话翻译
4. 其他的NLP炼丹技巧实践

其他参考实践

1. bert关系抽取：[Ricardokevins/Bert-In-Relation-Extraction: 使用Bert完成实体之间关系抽取 (github.com)](https://github.com/Ricardokevins/Bert-In-Relation-Extraction)
2. 文本语意匹配：[Ricardokevins/Text_Matching: NLP2020中兴捧月句子相似度匹配 (github.com)](https://github.com/Ricardokevins/Text_Matching)
3. Transfomer实现和其他部件：[Ricardokevins/EasyTransformer: Quick start with strong baseline of Bert and Transformer without pretrain (github.com)](https://github.com/Ricardokevins/EasyTransformer)

# TextClassification

文本分类demo，还同时作为baseline对比，加上了很多NLP trick实践

### 模型

1. bert-base-chinese
2. Base RNN
3. Base RNN + Attention
4. Base transformer

### 技术

1. 对抗训练
2. 模型蒸馏





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