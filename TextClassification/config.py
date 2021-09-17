Trainbert = False
USE_CUDA = True

num_epochs = 30
batch_size = 64
if Trainbert:
    batch_size = 16


# mode = 'train_with_FGM'
# mode = "train_Kd"
#mode = 'train'
mode = 'train_with_RDrop'
