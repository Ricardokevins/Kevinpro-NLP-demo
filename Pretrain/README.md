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



