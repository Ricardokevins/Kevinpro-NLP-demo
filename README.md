# NLP

个人的NLP实践demo。部分来源于其他开源项目。欢迎Star Fork。

主要内容

1. **文本分类,BiLSTM,Transfomer**
2. **摘要生成,Pointer Generator NetWork**
3. **对话翻译 Seq2Seq**
4. **GNN在文本分类的实践**
5. **Transformer Mask Language Model预训练**
6. **GPT文本续写以及GPT做数学题（偷的hhh）**
7. **其他的NLP炼丹技巧实践 对抗学习等**

其他参考实践

1. **bert关系抽取**：[Ricardokevins/Bert-In-Relation-Extraction: 使用Bert完成实体之间关系抽取 (github.com)](https://github.com/Ricardokevins/Bert-In-Relation-Extraction)
2. **文本语意匹配**：[Ricardokevins/Text_Matching: NLP2020中兴捧月句子相似度匹配 (github.com)](https://github.com/Ricardokevins/Text_Matching)
3. **Transfomer实现和其他部件**：[Ricardokevins/EasyTransformer: Quick start with strong baseline of Bert and Transformer without pretrain (github.com)](https://github.com/Ricardokevins/EasyTransformer)












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



## 2021.7.11

1. 增加了GNN在NLP中的应用
2. 实现了GNN在文本分类上的使用
3. 效果不好，暂时怀疑是数据处理的问题

## 2021.7.29

1. 增加了CHI+TFIDF传统机器学习算法在文本分类上的应用
2. 实现和测试了算法性能
3. 更新了README

## 2021.8.2

1. 重构了对话机器人模型于Seq2Seq文件夹
2. 实现了BeamSearch解码方式
3. 修复了PGN里的BeamSearch Bug



## 2021.9.11

1. 添加了GPT在文本续写和数学题问题的解决（偷了[karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training (github.com)](https://github.com/karpathy/minGPT)代码实现的很好，对理解GPT很有帮助，偷过来看看能不能用在好玩的东西
2. 重构了Pointer Generator NetWork，之前的表现一直不好，打算干脆重构，一行一行的重新捋一遍，感觉会安心很多。施工ing。



## 2021.9.16

1. 修复了Pretrain里Mask Token未对齐，位置不一致问题
