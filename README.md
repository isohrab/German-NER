# German-NER
Named Entity Recognition with German dataset by using BiLSTM and CNN in Tensorflow.

This Repo is based following articles:
 * Boosting Named Entity Recognition with Neural Character Embeddings by Cicero Nogueira dos Santos, Victor Guimarães.([PDF](http://www.anthology.aclweb.org/W/W15/W15-3904.pdf))
 * Named Entity Recognition with Bidirectional LSTM-CNNs by Jason P.C. Chiu, Eric Nichols.([PDF](https://arxiv.org/pdf/1511.08308.pdf))

### Input Data:
* A pre-trained word2vector for word embedding.
* A dataset consists of ca. 82000 sentences.
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
The whole dataset is divided into 10 parts. We take 8 parts, and two 1 part corresponding to the train-set, dev-set and the test set respectively. The train-set and dev-set are used to preprocessing as well as extracting all characters, words as and tags.

### Model:
The model consists of two main stages: Char level CNN and word level BiLSTM.
* ##### Char level CNN:
  First, we train character embedding with ```CHAR_EMB_DIM``` dimension with benefiting the ```tf.nn.embedding_lookup```. The shape of result looks like ```[batch_size * sentence_length , max_word_length , char_embedding_size]```. To feed it to a convolution layer we expand embedded chars to the 4D tensor by adding a new dimension to it. It's needed to provide a tensor with shape of ```[batches, height, width, channels]``` for the ```tf.nn.conv2d``` function, therefore the channel is equal to 1.
  Moreover, the filter has a shape of ```[filter_size, CHAR_EMB_DIM, 1, N_FILTERS]```. We use ReLU activation function after max pooling and concatenating all pooled outputs, and each word gets a vector with length of ```[len(FILTER_SIZE) * N_FILTERS)```.
  Finally, we concatenate this vector to a ```word_embedding``` vector.

* ##### Word level BiLSTM:
  Now, we have a tensor with shape of ```[Batch_size, Sentence_length, Word_embed+charCNN]```. We Initialize ```Forward_cell``` and ```backward_cell``` with ```HIDDEN_SIZE``` units. After running ```tf.nn.bidirectional_dynamic_rnn```, we get two final outputs for each forward and backward LSTMs and then concatenate them to a tensor with the shape of ```[Batch_Size , sentence_length, 2*HIDDEN_SIZE]```. Then, we use two fully connected layers with using a ReLU activation between. It's noticeable that using the second fully connected layer increases accuracy as well as F1 score 3% more.
  
### Hyper parameters and results:
* Learning rate and optimizer: In all training steps, we use Adam optimizer with learning rate of 0.001 and learning decay of 0.95.
* LSTM Hidden size: We try 100 and 400 units which 400 units clearly has a better performance.
* CHAR_EMB_DIM: We try 60, 120 and 200 which 120 is an optimum value.
* FILTER_SIZE: We used ```[2,3,4]``` as filter size of convolution layer.
* N_FILTERS: we use 32, 64 and 128 as number of filters which 128 has a better performance.
* DROPOUT: we use dropout 0.5 and presume that the model with 400 units can't be fit to the training data very well, but With dropout 0.75 we get a better result on F1 score and recall.
* MAX_LENGTH_WORD: To work with convolution layer we need to determine the maximum length of the word. The mean and maximum of unique words in the dataset are equal to 10 and 50 respectively. We use only max number in our experiments.
* BATCH_SIZE: 40
* N_EPOCHS: 100

##### Best Results:
  The best result we've got so far has been produced by the following layout:
  * Learning rate and optimizer: Adam, 0.001 and decay: 0.95
  * LSTM Hidden size: 400
  * CHAR_EMB_DIM: 120
  * FILTER_SIZE: [2, 3, 4, 5]
  * N_FILTERS: 128
  * DROPOUT: 0.5
  * MAX_LENGTH_WORD: 50
  * BATCH_SIZE: 40
  * N_EPOCHS: 50

We got accuracy: 96.25 %, F1 score: 68.73%, Precision: 68.02% and Recall: 69.48%.


### Acknowledgments 

Some parts of this code, borrowed or inspired by following resources. I'm appreciating for their efforts.
 * https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
 * http://cs231n.github.io/convolutional-networks/
 * http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

