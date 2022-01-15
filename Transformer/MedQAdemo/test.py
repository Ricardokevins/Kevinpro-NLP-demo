from model import Transformer
import torch
from trainer import DecodeData

@torch.no_grad()
def greedy_decoder(model,input):
    model.eval()
    model.cuda()
    decode_steps = 100
    decode_dataset = DecodeData()
    enc_input,dec_input = decode_dataset.encode(input)
    print(enc_input)
    enc_input = torch.tensor(enc_input).unsqueeze(0).cuda()
    dec_input = torch.tensor(dec_input).unsqueeze(0).cuda()


    for k in range(decode_steps):
        dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_input,dec_input)
        _, ix = torch.topk(dec_logits, k=1, dim=-1)
        print(ix)
        ix = ix[-1,:]
        ix = ix.reshape(1,-1)
        print(ix)
        dec_input = torch.cat((dec_input, ix), dim=1)
        #print(dec_input)

    result = dec_input.cpu().numpy().tolist()[0]
    tokens = decode_dataset.decode(result)

    output_string = "".join(tokens)
    output_string = output_string.replace("[PAD]","")
    output_string = output_string.replace("[BOS]","")
    output_string = output_string.replace("[EOS]","")
    print("Kevin'Transformer Doctor :",output_string)

model = Transformer()
model.load_state_dict(torch.load('./data/novelModel.pth'))


#input = "头疼头晕怎么办"
# 恶心想吐，不想吃饭咋办
question = "高血压患者能吃党参吗？,我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？"
#question = "恶心想吐，不想吃饭咋办"

#question = input("输入你要问医生的问题：")
greedy_decoder(model, question)