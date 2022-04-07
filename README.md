# NLP

**Some Simple implement of Fun NLP algorithm in Pytorch. updating and maintaining**

**If you have any question, please comment in Issue**

**If project helps you, welcome Star~ (Please Dont Just Fork without Star (´･ω･`) ）**

[中文版本README](https://github.com/Ricardokevins/Kevinpro-NLP-demo/blob/main/Chinese.md)

# The main content

you can go into each project folder for more details in folder's readme.md inside,  

1. **Text Classification Based on many Models (BiLSTM,Transformer)  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/TextClassification)**
2. **Summary Generation (Pointer Generator NetWork)  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/PGNSum)**
3. **Dialogue Translation (Seq2Seq) to build your own DialogueBot~~  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/ChatBotEnglish)**
4. **Use GNN in Text Classification  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/GNN)**
5. **Transformer Mask Language Model Pretraining  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/Pretrain)**
6. **GPT for Text Generation and GPT for math problem  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/GPT)**
7. **Adversarial training (FGM)  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/blob/main/TextClassification/Attack.py)**
8. **Very Simple and quick Use/Deploy of Seq2Seq-Transformer. Including Several Eamples(Denoise Pretrain, Medical-QuestionAnswering  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/Transformer)**
9. **Practical use of Pytorch_Lighting  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/blob/main/TextClassification/LightingMain.py)**
10. **AMP and Fp16 training for Pytorch  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/blob/main/VAEGenerator/transformerBased.py)**
11. **Usefully Visualize Toolkit for Attention Map(or Other weighted Matrix  [go here](https://github.com/Ricardokevins/Kevinpro-NLP-demo/tree/main/Visualize)**


My other open source NLP projects

1. **BERT in Relation Extraction**：[Ricardokevins/Bert-In-Relation-Extraction: 使用Bert完成实体之间关系抽取 (github.com)](https://github.com/Ricardokevins/Bert-In-Relation-Extraction)
2. **Text-matching**：[Ricardokevins/Text_Matching: NLP2020中兴捧月句子相似度匹配 (github.com)](https://github.com/Ricardokevins/Text_Matching)
3. **Transformer implement and useful NLP toolkit**：[Ricardokevins/EasyTransformer: Quick start with strong baseline of Bert and Transformer without pretrain (github.com)](https://github.com/Ricardokevins/EasyTransformer)

# What's New ~~
## 2022.3.25
1. Thanks to [@rattlesnakey](https://github.com/rattlesnakey)'s [Issue(more discussion detail here)](https://github.com/Ricardokevins/Kevinpro-NLP-demo/issues/15). I add Feature in Pretrain Project. Set the Attention Weight of MASK-Token to Zero to prevent MASK-Tokens Self-Attention Each other. You can enable this feature in Transformer.py by setting "self.pretrain=True". PS:The New Feature has not been verified for the time being, and the effect on the pre-training has not been verified. I'll fill in the tests later
## 2022.1.28
1. Rebuild the code structure in Transformer. Make Code Easier to Use and deploy
2. Add Examples: Denoise-Pretrain in Transformer (Easy to use)
## 2022.1.16
1. Update use Seq2Seq Transformer to Modeling Medical QA task   (Tuing on 55w pairs of Chinese Medical QA data) More detail to be seen in README.md of Transformer/MedQAdemo/
2. Update new Trainer and useful tools
3. remove previous implement of Transformer (with some unfixable bugs)
## 2021.12.17

1. Update Weighted Matrix Visualize Toolkit(eg. used for visualize of Attention Map) implement in Visualize. More Useful toolkit in the future
2. Update Python comment Code Standards. More formal code practices will be followed in the future.

## 2021.12.10

1. Update Practical use of Pytorch_Lighting, Use Text_classification as Example. Convert the Pytorch to LightningLite. More details in LightingMain.py。
2. Remove the redundant code

## 2021.12.9

1. update Practical use of Amp(Automatic Mixed Precision). Implement in VAEGenerator, Test on local MX150, Significant improve the training time and  Memory-Usage, More details in Comments at the end of the code
2. Based the command of Amp, Modified the definition of 1e-9 to inf in model.py



# Update History

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

## 2021.9.29

1. 在Transformer里增加了一个随机数字串恢复的Demo，对新手理解Transformer超友好，不需要外部数据，利用随机构造的数字串训练
2. 新增实验TransfomerVAE，暂时有BUG，施工中

## 2021.11.20

1. update BM25 and TF-IDF algorithm for quick match of Text.
# 参考

## BM25

<https://blog.csdn.net/chaojianmo/article/details/105143657>

## Automatic Mixed Precision (AMP)

<https://featurize.cn/notebooks/368cbc81-2b27-4036-98a1-d77589b1f0c4>
