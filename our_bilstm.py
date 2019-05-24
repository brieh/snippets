
# coding: utf-8

# Ideas for improving training, increase drop out? add batch normalization? increase hidden dimensions because of big vocab?

# In[2]:


# Code modified from: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
from torchtext import data
import torch
import torch.optim as optim
import torch.nn as nn


import random
import matplotlib.pyplot as plt
import time


import sys
sys.stdout = open("log2.txt", "w")

# In[3]:


SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[4]:


# include_lengths this time so we can tell our network how long each sequence actually 
# is, which will allow it to only process the non-padded elements of each sequence.
# output for pad elements will be a zero tensor

TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)


# In[5]:


# Load our preprocessed dataset
path_to_train = r'data/train.csv'
path_to_test = r'data/test.csv'
fmt = 'CSV'

fields=[('IDLink', None),        	# ignores the IDLink
        ('Title', None),         	# ignores Title column
        ('Headline', None),		# ignores the Headline column
        ('Source', None),		# ignores Source
        ('Topic', None),       		# ignores Topic
        ('PublishDate', None),      	# ignores PublishDate
        ('SentimentTitle', None),   	# ignores SentimentTitle
        ('SentimentHeadline', None),	# ignores SentimentHeadline
        ('Facebook', None),      	# ignores the Facebook column
        ('GooglePlus', None),      	# ignores the GooglePlus column
        ('LinkedIn', None),      	# ignores the LinkedIn column
        ('label', LABEL),        # imports the label column as LABEL
        ('title_pp', None),      	# ignores the title_pp column
        ('text', TEXT)]          # imports the headline_pp column as our TEXT 


# In[6]:


train_data = data.TabularDataset(path_to_train, fmt, fields, skip_header=True)
test_data = data.TabularDataset(path_to_test, fmt, fields, skip_header=True)

# split to train, valid and test sets
train_data, valid_data = train_data.split(split_ratio=[0.7,0.3])


# In[7]:


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(vars(train_data.examples[0]))


# In[8]:


# Build the Vocabularies
TEXT.build_vocab(train_data
                 #, max_size=25000
                 , vectors="glove.6B.300d"
                 , unk_init = torch.Tensor.normal_)      # initializes vocabulary words not found 
                                                         # in pre-trained embeddings to a normalized
                                                         # Gaussian (instead of zero)
    
LABEL.build_vocab(train_data)


# In[9]:


print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)


# In[10]:


# Creating the Iterator
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data)
    , batch_size=BATCH_SIZE
    , sort_key=lambda x: len(x.text)    
    , sort_within_batch=True
    , device=device)
# for packed padded sequences, we need to sort all tensors in a batch by length


# In[11]:


# Create and Train a model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim
                           , hidden_dim
                           , num_layers=n_layers
                           , bidirectional=bidirectional
                           , dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # use drop out to combat overfitting
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden.squeeze(0))





# In[32]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


# In[33]:


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    db = 0

    if db == 1:
        c = 0

    for batch in iterator:
        if db == 1:
            print('{}%'.format(c/len(train_iterator)*100))
            c += 1
            
        optimizer.zero_grad()
        
        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        
        for batch in iterator:
            
            text, text_lengths = batch.text
        
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def grid_train_model(HIDDEN_DIM, N_LAYERS):


    print('For {} hidden dimension and {} layers'.format(HIDDEN_DIM, N_LAYERS))

    # For BiLSTM
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300      # embedding dim must match length of pretrained glove vectors
    OUTPUT_DIM = 1
    BIDIRECTIONAL = True
    DROPOUT = 0.85

    # Get pad token index from the vocab using stoi method with the string representing 
    # the pad token which we get from the field's pad_token attribute, which is <pad> by default.
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



    model = BiLSTM(INPUT_DIM
                   , EMBEDDING_DIM
                   , HIDDEN_DIM
                   , OUTPUT_DIM
                   , N_LAYERS
                   , BIDIRECTIONAL
                   , DROPOUT
                   , PAD_IDX)



    # Initialize the weights in the embedding layer with the pretrained embeddings
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)



    # We set the embeddings for our <unk> and <pad> tokens to zeros

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)



    N_EPOCHS = 30

    train_losses = []
    val_losses = []

    best_valid_loss = float('inf')


    for epoch in range(N_EPOCHS):
        
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        train_losses.append(train_loss)
        
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        val_losses.append(valid_loss)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # keep track of the best model as we go along
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        

    # do a plot of train loss and validation loss 
        
    #plt.plot(train_losses, label='train')
    #plt.plot(val_losses, label='validation')
    #plt.title("training loss & validation loss")
    #plt.legend()
    #plt.show()    
    #plt.savefig("train_val_loss_{}_{}.png".format(HIDDEN_DIM, N_LAYERS))


    # load the best model from this training run and use it to evaluate the test set
    model.load_state_dict(torch.load('tut2-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')







for h in range(300,1000,50):
    for nl in range(2,6): 
        grid_train_model(h,nl)

