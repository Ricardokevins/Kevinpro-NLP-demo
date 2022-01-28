# Simple Transformer Implement And Deploy

## How To Use
There are 2 Examples in DenoisePT and MedQAdemo
DenoisePT is Seq2Seq-Transformer Pretrain(like BART)
MedQAdemo is Chinese-Medical Question-Answering

1. Create Folder "xxxx" and Create Folder "xxx/data"
2. Implement your "prepare_data.py" to generate "source.txt" and "target.txt"
3. Follow 2 examples to implement your "main.py"
   1. if Needed, you can define your DataConfig to control Dataset Size and MaxLength of Sequence
   2. you can also define your own Training Config in main.py
4. Enjoy Happy Training Time ~