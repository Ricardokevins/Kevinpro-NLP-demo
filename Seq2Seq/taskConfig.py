
padding_source_length = 200
padding_target_length = 100

attn_model = 'dot'
 
max_vocab = 50000
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 300
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

optim = 'AdamW'

lr = 2e-5

eps = 1e-8
epoch = 20

teacher_forcing_ratio = 0.5

decoder_learning_ratio = 5.0

clip = 50.0
