# German-NER
Named Entity Recognition with German Dataset by using BiLSTM and CNN in Tensorflow

#### Input Data:
* A pre-trained word2vector for word embedding.
* A dataset consist of ca. 82000 sentences.
```csv
1    Verlag    _    _    _    B-ORG    _    _    _    _
2    Paul    _    _    _    I-ORG    _    _    _    _
3    Haupt    _    _    _    I-ORG    _    _    _    _
4    ,    _    _    _    O    _    _    _    _
5    Bern    _    _    _    B-LOC    _    _    _    _
6    und    _    _    _    O    _    _    _    _
7    Stuttgart    _    _    _    B-LOC    _    _    _    _
8    ,    _    _    _    O    _    _    _    _
9    1999    _    _    _    O    _    _    _    _
10    (    _    _    _    O    _    _    _    _
11    gemeinsam    _    _    _    O    _    _    _    _
12    mit    _    _    _    O    _    _    _    _
13    P.    _    _    _    B-PER    _    _    _    _
14    Rusterholz    _    _    _    I-PER    _    _    _    _
15    )    _    _    _    O    _    _    _    _
16    .    _    _    _    O    _    _    _    _
```
The whole dataset is divided into 10 part, which 8 parts is used for training, 1 part for devset and 1 part for the test. The trainset and devset are used for preprocessing to extract all chars, words, and tags.

#### Model:
The model is consist of two main stage, Char level CNN and word level BiLSTM.
* ##### Char level CNN:
  First, we train char embedding with ```CHAR_EMB_DIM``` dimension with help of ```tf.nn.embedding_lookup```. The result has a shape of ```[batch_size * sentence_length , max_word_length , char_embedding_size]```. To feed this to Convolution layer we expand embedded Char to 4D by adding a Dimension at the end of the tensor. We need to do this to provide a tensor with shape ```[batches, height, width, channels]``` for ```tf.nn.conv2d``` function. So here channel is 1.
  The Filter has a shape of ```[filter_size, CHAR_EMB_DIM, 1, N_FILTERS]```. We used ReLU activation function after  After maxpooling and concatenate all pooled outputs, each word will get a vector of length ```[len(FILTER_SIZE) * N_FILTERS).
  Finally, we concatenate this vector to word_embedding vector.

* ##### Word level BiLSTM:
  Now, we have a tensor with shape of ```[Batch_size, Sentence_length, Word_embed+charCNN]```. We Initialized ```Forward_cell``` and ```backward_cell``` with ```HIDDEN_SIZE``` units. After running ```tf.nn.bidirectional_dynamic_rnn```, we get two final outputs for each forward and backward LSTMs and then concatenate them to a tensor with shape of ```[Batch_Size , sentence_length, 2*HIDDEN_SIZE].
  After that, we used two fully connected layers which a ReLU activation in between is used. We should emphasize that by using second fully connected layer, we could increase accuracy  F1 score by 3%.
  
#### Hyperparameters and results:
* Learning rate and optimizer: In all training, we used Adam optimizer with a Learning rate of 0.001 and learning decay of 0.95.
* LSTM Hidden size: We tried 100 and 400 units which 400 units clearly had better performance.
* CHAR_EMB_DIM: We tried 60, 120 and 200 which 120 was an optimum value.
* FILTER_SIZE: We used ```[2,3,4]``` sizes as filter sizes for convolution layer.
* N_FILTERS: we used 32, 64 and 128 sizes which we saw 128 had better performance.
* DROPOUT: we tried dropout with 0.5 probability which I think the model with 400 hidden layers could not fit training data very well, with dropout=0.75, we could better result on F1 score and recall.
* MAX_LENGTH_WORD: To work with convolution layer we need to determine the maximum length of the word. The mean of unique words in the dataset is 10, and max is 50. We used only max number in our experiments.
* BATCH_SIZE: 40
* N_EPOCHS: 100

##### Best Results:
  The best results so far that I write this readme, with follwoing config:
  * Learning rate and optimizer: Adam, 0.001 and decay: 0.95
  * LSTM Hidden size: 400
  * CHAR_EMB_DIM: 120
  * FILTER_SIZE: [2,3,4]
  * N_FILTERS: 128
  * DROPOUT: 0.75
  * MAX_LENGTH_WORD: 50
  * BATCH_SIZE: 40
  * N_EPOCHS: 100
  * BATCH_SIZE: 40
  We got accuracy: 96.14 %, F1 score: 66.93, Precision: 67.02 and Recall: 66.84.
