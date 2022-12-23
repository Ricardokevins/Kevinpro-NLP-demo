from chatgpt_wrapper import ChatGPT
bot = ChatGPT()

path = "/Users/sheshuaijie/Downloads/sampled_data_30(1).json"
f = open(path,'r')
import json
data = json.load(f)
#print(data[0])
from tqdm import tqdm
for i in tqdm(data):
    dialogue = i['Dialogue']
    response = bot.ask("Convert the following conversation to 3rd person POV " + dialogue)
    i['Shifted_Dialogue'] = response

with open('sampled_data_30(1)_shifted.json', 'w') as outfile:
    json.dump(data, outfile,ensure_ascii=False,indent=2)





exit()
# dialogue = "Lenny: Babe, can you help me with something?\r\nBob: Sure, what's up?\r\nLenny: Which one should I pick?\r\nBob: Send me photos\r\nLenny:  <file_photo>\r\nLenny:  <file_photo>\r\nLenny:  <file_photo>\r\nBob: I like the first ones best\r\nLenny: But I already have purple trousers. Does it make sense to have two pairs?\r\nBob: I have four black pairs :D :D\r\nLenny: yeah, but shouldn't I pick a different color?\r\nBob: what matters is what you'll give you the most outfit options\r\nLenny: So I guess I'll buy the first or the third pair then\r\nBob: Pick the best quality then\r\nLenny: ur right, thx\r\nBob: no prob :)"



# response = bot.ask("Convert the following conversation to 3rd person POV " + dialogue )
# print(response)  # prints the response from chatGPT