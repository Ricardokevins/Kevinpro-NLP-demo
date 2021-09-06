from Baseline import EncoderRNN
from Baseline import Attn
from Baseline import LuongAttnDecoderRNN
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from dataloader import loadPrepareData
from dataloader import indexesFromSentence
import os
checkpoint_iter = 4000
save_dir = os.path.join("data", "save")
model_name = 'cb_model' 
attn_model = 'dot'
hidden_size = 256
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子
setup_seed(20)
MIN_COUNT = 3    # Minimum word count threshold for trimming
MAX_LENGTH = 10  # Maximum sentence length to consider
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
corpus_name = ""
corpus = os.path.join("data", corpus_name)
save_dir = os.path.join("data", "save")
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    print(loadFilename)
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.eval()
decoder.eval()

class beam_state:
    def __init__(self, decoder_hidden, decoder_input, tokens, score):
        self.decoder_hidden, self.decoder_input, self.tokens,self.score = decoder_hidden, decoder_input, tokens, score

    @property
    def avg_log_prob(self):
        #print("count",self.score)
        return sum(self.score) / len(self.tokens)

def sort_beams(beams):
    return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

class BeamSearchDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beamsize = 4
    
    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        encoder_outputs = torch.cat([encoder_outputs for i in range(self.beamsize)], 1)
        #encoder_hidden = torch.cat([encoder_hidden for i in range(self.beamsize)], 1)

  
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        # all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        # all_scores = torch.zeros([0], device=device)
        all_scores = []
        all_tokens = []
        beam_list = [beam_state(decoder_hidden, decoder_input,all_tokens ,all_scores) for i in range(self.beamsize)]
        from tqdm import tqdm
        for _ in range(max_length):
            all_decoder_hidden =[]
            all_decoder_input = []
 
            for i in range(self.beamsize):
                all_decoder_hidden.append(beam_list[i].decoder_hidden)
                all_decoder_input.append(beam_list[i].decoder_input)
            
            decoder_hidden_input = torch.cat(all_decoder_hidden, 1)
            decoder_input_input = torch.cat(all_decoder_input, 1)

            #print(decoder_hidden_input.shape,decoder_input_input.shape,encoder_outputs.shape)
            decoder_output_all, decoder_hidden_all = self.decoder(decoder_input_input, decoder_hidden_input, encoder_outputs)

            topk_probs, topk_ids = torch.topk(decoder_output_all, self.beamsize,dim=1)

            all_beams = []
            num_orig_beams = 1 if _ == 0 else len(beam_list)
            for i in range(num_orig_beams):
                beam_states = beam_list[i]
                for j in range(self.beamsize):
                    new_score = beam_states.score.copy()
                    #print("New,Score1",new_score)
                    new_score.append(topk_probs[i,j].item())
                    #print("New,Score2",new_score)
                    new_token = beam_states.tokens.copy()
                    new_token.append(topk_ids[i, j].item())
                    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * topk_ids[i, j].item()
                    decoder_hidden = decoder_hidden_all[:,j,:]
                    decoder_hidden = decoder_hidden.reshape(decoder_hidden.shape[0],1,-1)
                    New_Beam_State = beam_state(decoder_hidden, decoder_input, new_token, new_score)
                    all_beams.append(New_Beam_State)
            # print("------------------------------")
            # for i in all_beams:
            #     print(i.tokens)
            #     print(i.score)
            beams = []
            for h in sort_beams(all_beams):
                beams.append(h)
                if len(beams) == self.beamsize:
                    break
            
            beam_list = beams
        beams_sorted = sort_beams(beams)
        #print(beams_sorted[0].tokens)
        #print(beams_sorted[0].score)
        return torch.tensor(beams_sorted[0].tokens),torch.tensor(beams_sorted[0].score)

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInside(encoder, decoder, searcher, voc,testset):
    from tqdm import tqdm 
    from rouge import Rouge 
    predicts = []
    targets = []
    sources = []
    for test in tqdm(testset):
        try:
            source = test[0]
            target = test[1]
            sources.append(source)
            input_sentence = source
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            predict = ' '.join(output_words)
            predicts.append(predict)
            targets.append(target)
        except KeyError:
            pass

    rouge = Rouge()
    scores = rouge.get_scores(predicts, targets, avg=True)
    print(len(predicts))
    print(scores)
    f = open('greedy_results.txt', 'w',encoding  = 'utf-8')
    for t,i,j in zip(sources,predicts,targets):
        f.write(t+'\n')
        f.write(i+"     "+j+'\n')
        f.write("==================================================\n")
    f.close()

def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            #input_sentence = "how are you doing man"
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': 
                break
            # Normalize sentence
            #input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


searcher = GreedySearchDecoder(encoder, decoder)
#searcher = BeamSearchDecoder(encoder, decoder)

def BotAPI(input_sentence):
    try:
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    except KeyError:
        output_words = ['unable to answer']
    return 'Bot: '+' '.join(output_words)

def test():
    from dataloader import trimRareWords
    pairs2 = trimRareWords(voc, pairs, MIN_COUNT)

    f = open("TestSet.txt",'w',encoding='utf-8')

    test_set_size = 1000
    import random

    test_set = []
    while len(test_set) < test_set_size:
        test_sample = random.choice(pairs2)
        if test_sample not in test_set:
            test_set.append(test_sample)

    test_set = [test_set[0]]
    print(test_set[0])
    #searcher = BeamSearchDecoder(encoder, decoder)
    evaluateInside(encoder, decoder, searcher, voc,test_set)

#test()


#{'rouge-1': {'f': 0.08770590099211, 'p': 0.09842976190476202, 'r': 0.0928503968253969}, 'rouge-2': {'f': 0.015368631185338202, 'p': 0.016616666666666665, 'r': 0.016145238095238094}, 'rouge-l': {'f': 0.08617159363142878, 'p': 0.09672976190476201, 'r': 0.0910988095238096}}
#{'rouge-1': {'f': 0.09156701700941122, 'p': 0.1092472222222222, 'r': 0.09344126984126996}, 'rouge-2': {'f': 0.014569763394207222, 'p': 0.017352380952380956, 'r': 0.014278571428571427}, 'rouge-l': {'f': 0.091432549840759, 'p': 0.10849960317460312, 'r': 0.09379880952380965}}