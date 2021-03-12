# Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction

This guide explains the main ideas in the paper by Hu and others 2019. For the code, the one given on the list was for the Statsnet. We assumed that we use the dataset provided (tweets and historical price data) with the Statsnet. The source code for the paper was not available from the authors. Thus, we used the following code:

https://github.com/donghyeonk/han

This implementation does not include the Self-paced Learning mechanism, which will be dscribed shortly. This implementation also does not include the stock prediction strategy based on the classification of stock trends. For these sections, we refer to the original paper. 

## Introduction

Stock trend prediction is an interesting problem with many applications in finance. For instance, understanding how individual stocks trend will allow for maximizing profit from the stock investment (Hu et al. 2019). To this end, the authors propose a method based on the human learning processes: sequential context dependency, diverse influence, and effective and efficient learning. 

Sequential context dependency refers to the ability of human investors to implement and combine a variety of news sources on a single stock into a unified context before making a decision. Thus, to mimic such human analytical process, the authors propose that the prediction framework should incorporate the news sources in a sequential, temporal context. 

Diverse influence indicates that some news sources have more influential and durable effects on the stocks' trends. Thus, these news sources should have more attention paid to them than those that do not have such characteristics. 

Effective and efficient learning indicates that human investors tend to first gain an overall knowledge with common occasions, and then turn to exceptional cases (Hu et al. 2019). Thus a learning mechanism should incorporate such discrepancies in the informativeness of the news sources, and conducts learning on more informative news at the earlier stage, and further optimize to tackle harder samples (Hu et al. 2019). 

The guide is structured as follows: we first describe how the first two strategies are incorporated into the Hybrid Attention Network (HAN) proposed by the authors. We describe the self-paced learning mechanism. Then, we describe the learning of the network on the Statsnet dataset, with a focus on the description of the data and the learning curves. Lastly, we describe the stock trading strategy based on stock trend classifications as proposed by the authors. 

## HAN - Hybrid Attention Network

Hybrid Attention Network (HAN), as proposed by the authors, have the ability to predict a stock's trend as a classification problem. For the Statsnet dataset, the classification problem can be divided into the following two classes: DOWN, UP. To classify into these two classes, the rise percentage at a given time point is calculated, where rise percent(t) = [open price(t+1) - open price(t)]/ open price(t).

The Sequential context dependency, and the Diverse influence principles indicate that 1) the news are analyzed in a temporal fashion, with more attention to critical times, and 2) news are distinguished by significance. Thus, HAN incorporates the attention mechanisms at both the news level and and temporal level. The following figure from the paper summarizes the network architecture of HAN. 

![Alt text](/src/han.png?raw=true)

### News Embedding

The authors use a word embedding layer to calculate the embedded vector for each word and then average all the words' vectors to construct a news vector. A Word2Vec is used as the word embedding layer. 

### News-level Attention

To implement the Diverse influence mechanism proposed, an attention mechanism is introduced to aggregate the news weighted by an assigned attention value in order to reward the news offering critical information. First, the attention values are estimated by feeding the news vector n(ti) through a one-layer network to get the news-level attention value u(ti) (Eq. 1). The attention weights are normalized through a softmax function (Eq. 2). Finally, the overall corpus vector is calculated as a weighted sum of each news vector (Eq. 3) and is used to represent all news information at date t. 

<img src="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;u_{ti}&space;&=&space;\text{sigmoid}(W_nn_{ti}&space;&plus;&space;b_n),&space;\\&space;\alpha_{ti}&space;&=&space;\frac{\text{exp}(u_{ti})}{\sum_j&space;\text{exp}(u_{tj})},&space;\\&space;d_{t}&space;&=&space;\sum_{i}\alpha_{ti}n_{ti}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;u_{ti}&space;&=&space;\text{sigmoid}(W_nn_{ti}&space;&plus;&space;b_n),&space;\\&space;\alpha_{ti}&space;&=&space;\frac{\text{exp}(u_{ti})}{\sum_j&space;\text{exp}(u_{tj})},&space;\\&space;d_{t}&space;&=&space;\sum_{i}\alpha_{ti}n_{ti}&space;\end{align*}" title="\begin{align*} u_{ti} &= \text{sigmoid}(W_nn_{ti} + b_n), \\ \alpha_{ti} &= \frac{\text{exp}(u_{ti})}{\sum_j \text{exp}(u_{tj})}, \\ d_{t} &= \sum_{i}\alpha_{ti}n_{ti} \end{align*}" /></a>

### Sequential Modeling

To encode the temporal sequence of corpus vectors, the authors adopt Gated Recurrent Units (GRU). For the current news state at date t, the GRU computes it by linearly interpolating the previous state and the current updated state. For the current updated state, it is computed by non-linearly combining the corpus vector input for date t and the previous state with the reset gate and the update gate. Finally, the latent vectors from both directions are concatenated to construct a bi-directional encoded state vector h(t). 

### Temporal Attention

The temporal-level attention mechanism is used to incorporate Sequential context dependency principle into HAN. Similar to the attention mechanim at the news-level, the temporal-level attention mechanim also computes the latent representations of encoded corpus vectors. By combining them through a softmax layer, an attention vector beta is calculated, and the weighted sum is calculated to incorporate the sequential news context information with temporal attention (V). 

<img src="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;o_{i}&space;&=&space;\text{sigmoid}(W_hh_{i}&space;&plus;&space;b_h),&space;\\&space;\beta_{i}&space;&=&space;\frac{\text{exp}(\theta_io_i)}{\sum_j\text{exp}(\theta_io_i)},&space;\\&space;V&space;&=&space;\sum_i&space;\beta_ih_i&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;o_{i}&space;&=&space;\text{sigmoid}(W_hh_{i}&space;&plus;&space;b_h),&space;\\&space;\beta_{i}&space;&=&space;\frac{\text{exp}(\theta_io_i)}{\sum_j\text{exp}(\theta_io_i)},&space;\\&space;V&space;&=&space;\sum_i&space;\beta_ih_i&space;\end{align*}" title="\begin{align*} o_{i} &= \text{sigmoid}(W_hh_{i} + b_h), \\ \beta_{i} &= \frac{\text{exp}(\theta_io_i)}{\sum_j\text{exp}(\theta_io_i)}, \\ V &= \sum_i \beta_ih_i \end{align*}" /> </a>

### Trend Prediction

Finally, a standard MLP is used to take input V and produce the two-class classification.

## Self-paced Learning Mechanism

The authors take advantage of the Self-Paced Learning (SLP) to embed curriculum design into the learning objective to learn the news influence in an organized manner. For a gven learning sample, an importance weighte v(i) is assigned. Then, the goal of the SLP is to learn the model parameter and the laten weight jointly:

<img src = "https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\text{min}_{w,v&space;\in&space;[0,1]^n}&space;E(w,v,\lambda)&space;&=&space;\sum_{i=1}^{n}v_iL(y_i,HAN(x_i,w))&space;&plus;&space;f(v;\lambda),&space;\\&space;f(v;\lambda)&space;&=&space;\frac{1}{2}\lambda\sum_{i=1}^n(v_i^2&space;-&space;2v_i)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\text{min}_{w,v&space;\in&space;[0,1]^n}&space;E(w,v,\lambda)&space;&=&space;\sum_{i=1}^{n}v_iL(y_i,HAN(x_i,w))&space;&plus;&space;f(v;\lambda),&space;\\&space;f(v;\lambda)&space;&=&space;\frac{1}{2}\lambda\sum_{i=1}^n(v_i^2&space;-&space;2v_i)&space;\end{align*}" title="\begin{align*} \text{min}_{w,v \in [0,1]^n} E(w,v,\lambda) &= \sum_{i=1}^{n}v_iL(y_i,HAN(x_i,w)) + f(v;\lambda), \\ f(v;\lambda) &= \frac{1}{2}\lambda\sum_{i=1}^n(v_i^2 - 2v_i) \end{align*}" /> </a>

where L is the loss and f is the linear regularizer that discriminates samples with respect to their loss. 

As compared to the standard training process, the SLP takes longer in the beginning to learn the data, but has better accuracy results in the longer run. 

## Data and Implementation

The Stocknet dataset included in this experiment is the two-year price movements from 01/01/2014 to 01/01/2016 of 88 stocks, coming from all the 8 stocks in the Conglomerates sector and the top 10 stocks in capital size in each of the other 8 sectors. For the news component, the preprocessed tweet data is used, where the keys are 'text', 'user_id_str', and 'created_at'. For the preprocessed price data, the entries are: date, movement percent, open price, high price, low price, close price, volume. 

This dataset comprises two main components

- ./tweet: tweet data from Twitter
- ./price: price data from Yahoo Finance

Each component contains their raw data and preprocessed data organized by stocks.

We briefly describe the codes for data processing, HAN, and training and testing.

### Data Processing 

First, we load the stock history based on the historical price data. Since the dataset describes the two-year price movements from 2014 to 2016 of 88 stocks, we first make sure that we can calculate the rise percentages of the individual stocks, as defined. We then define a stock name dictionary, which contains the name of the stocks as keys and the rise percentage and the number of days that the stock has been in the dataset. Finally, we set the two classes bounds as follows:

- down_bound = -0.5  
- up_bound = 0.55 

Based on the stock name dictionary, we load the preprocessed tweets data. The preprocessed tweets data contain the key words of the tweets. We match the stock names to the tweets pertaining to those stocks, and save the lengths of the tweets and the contents of the tweets. Finally, the number of total samples with splits of train, validation, and test data is as follows: 

- 20345 total number of train samples
- 2555 number of validation samples
- 3720 test samples

The two classes are evenly divided for each of the samples. 

``` python

    def load_stock_history(self):

        # 0 date, 1 movement percent, 2 open price,
        # 3 high price, 4 low price, 5 close price, 6 volume

        # stock_dict
        # key: stock_name
        # val: [stock_name + '\t' + stock_date, close_price diff. percent]
        stock_dict = dict()
        diff_percentages = list()

        num_trading_days = 0

        file_names = os.listdir(self.flags.data_dir)
        for filename in file_names:
            stock_name = os.path.splitext(os.path.basename(filename))[0]

            if stock_name not in self.stock_name_set:
                continue

            if len(self.flags.whitelist) > 0 \
                    and stock_name not in self.flags.whitelist:
                continue

            filepath = os.path.join(self.flags.data_dir, filename)

            # trading day -1
            with open(filepath, 'r', encoding='utf-8') as f:

                # *reversed*
                for l in reversed(list(f)):
                    row = l.rstrip().split('\t')

                    stock_date = datetime.strptime(row[0], '%Y-%m-%d').date()

                    if not (date(2014, 1, 1) <= stock_date < date(2016, 1, 1)):
                        continue

                    price_diff_percent = float(row[1])

                    if stock_name not in stock_dict:
                        stock_dict[stock_name] = list()
                    stock_dict[stock_name].append(
                        [stock_date, price_diff_percent]
                    )

                    num_trading_days += 1

                    if len(stock_dict[stock_name]) > self.flags.days:
                        diff_percentages.append(price_diff_percent)

        num_ex = 0
        for stock_name in stock_dict:
            num_ex += len(stock_dict[stock_name]) - self.flags.days

        print('target stock history len', num_ex)
        print('num_trading_days', num_trading_days)

        down_bound = -0.5  # StockNet
        up_bound = 0.55  # StockNet

        return stock_dict, down_bound, up_bound
        
  
   def map_stocks_tweets(self):
   
        # StockNet
        train_x = list()
        train_y = list()
        dev_x = list()
        dev_y = list()
        test_x = list()
        test_y = list()

        train_lable_freq_dict = dict()
        dev_lable_freq_dict = dict()
        test_lable_freq_dict = dict()

        diff_percentages = list()

        num_dates = 0
        num_tweets = 0
        zero_tweet_days = 0
        num_filtered_samples = 0  # StockNet: no tweet lags

        for stock_name in self.stock_dict:

            stock_history = self.stock_dict[stock_name]

            stock_days = len(stock_history)

            num_stock_dates = 0
            num_stock_tweets = 0
            stock_zero_tweet_days = 0

            for i in range(stock_days):

                # StockNet
                if -0.005 <= stock_history[i][1] < 0.0055:
                    num_filtered_samples += 1
                    continue

                stock_date = stock_history[i][0]

                ex = list()
                day_lens = list()
                news_lens = list()

                days = list()

                num_empty_tweet_days = 0

                for j in [5, 4, 3, 2, 1]:
                    tweet_date = stock_date - timedelta(days=j)

                    stock_key = stock_name + '\t' + str(tweet_date)

                    ex_1 = list()
                    t_lens = list()

                    if stock_key in self.date_tweets:
                        tweets = self.date_tweets[stock_key]

                        for w_idxes in tweets:
                            ex_1.append(
                                '\t'.join([str(widx) for widx in w_idxes]))
                            t_lens.append(len(w_idxes))

                        day_lens.append(len(tweets))

                        num_stock_tweets += len(tweets)

                        if len(tweets) == 0:
                            num_empty_tweet_days += 1
                        else:
                            days.append(tweet_date)

                    else:
                        # no tweets date
                        day_lens.append(0)

                    ex.append('\n'.join(ex_1))
                    news_lens.append(t_lens)

                # StockNet: at least one tweet
                if num_empty_tweet_days > 0:
                    num_filtered_samples += 1
                    continue

                # StockNet
                if stock_history[i][1] <= 1e-7:
                    label = 0
                else:
                    label = 1

                label_date = stock_history[i][0]

                # split to train/dev/test sets
                if date(2014, 1, 1) <= label_date < date(2015, 8, 1):
                    train_x.append(ex)
                    train_y.append(label)

                    if label in train_lable_freq_dict:
                        train_lable_freq_dict[label] += 1
                    else:
                        train_lable_freq_dict[label] = 1

                    num_dates += self.flags.days
                    num_stock_dates += self.flags.days

                elif date(2015, 8, 1) <= label_date < date(2015, 10, 1):
                    dev_x.append(ex)
                    dev_y.append(label)

                    if label in dev_lable_freq_dict:
                        dev_lable_freq_dict[label] += 1
                    else:
                        dev_lable_freq_dict[label] = 1

                    num_dates += self.flags.days
                    num_stock_dates += self.flags.days

                elif date(2015, 10, 1) <= label_date < date(2016, 1, 1):
                    test_x.append(ex)
                    test_y.append(label)

                    if label in test_lable_freq_dict:
                        test_lable_freq_dict[label] += 1
                    else:
                        test_lable_freq_dict[label] = 1

                    num_dates += self.flags.days
                    num_stock_dates += self.flags.days

                else:
                    num_filtered_samples += 1
                    continue

                diff_percentages.append(stock_history[i][1])

            if num_stock_dates > 0:
                print(stock_name + '\t{:.2f}\t{}/{}\t{:.2f}\t{}/{}'.format(
                          num_stock_tweets / num_stock_dates,
                          num_stock_tweets, num_stock_dates,
                          stock_zero_tweet_days / num_stock_dates,
                          stock_zero_tweet_days, num_stock_dates))
            else:
                print(stock_name, 'no valid')

        return train_x, train_y, dev_x, dev_y, test_x, test_y
            
```
### Model

As discussed above, the model is 


```python
import tensorflow as tf


class HAN(tf.keras.Model):
    def __init__(self, wordvec, flags_obj):
        super(HAN, self).__init__()

        self.flags = flags_obj

        # wordvec: ndarray
        vocab_size = wordvec.shape[0]
        self.embedding_dim = wordvec.shape[1]
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, self.embedding_dim, weights=[wordvec], mask_zero=True,
            trainable=True,
            # embeddings_regularizer=tf.keras.regularizers.l2(self.flags.l2)
        )

        self.dropout = tf.keras.layers.Dropout(rate=self.flags.dr)  # StockNet

        # Word-level attention
        self.t = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # News-level attention
        self.u = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # Sequence modeling
        self.bi_gru = self.get_bi_gru(self.embedding_dim)

        # Temporal attention
        self.o = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # Discriminative Network (MLP)
        self.fc0 = tf.keras.layers.Dense(
            self.flags.hidden_size, activation=tf.nn.elu)
        self.fc1 = tf.keras.layers.Dense(
            self.flags.hidden_size, activation=tf.nn.elu)

        # StockNet: 2-class
        self.fc_out = tf.keras.layers.Dense(2)

    def call(self, x, day_len, news_len, training=False):
        max_dlen = tf.keras.backend.max(day_len).numpy()
        max_nlen = tf.keras.backend.max(news_len).numpy()
        x = x[:, :, :max_dlen, :max_nlen]
        news_len = news_len[:, :, :max_dlen]

        # Averaged daily news corpus
        # (batch_size, days, max_daily_news, max_news_words
        # -> (batch_size, days, max_daily_news, max_news_words, embedding_dim)
        x = self.embedding(x)

        # handle variable-length news word sequences
        mask = tf.sequence_mask(news_len, maxlen=max_nlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=4)
        x *= mask

        # Word-level attention
        # x: (batch_size, days, max_daily_news, max_news_words, embedding_dim)
        # t: (batch_size, days, max_daily_news, max_news_words, 1)
        # n: (batch_size, days, max_daily_news, embedding_dim)
        t = self.t(x)
        n = tf.nn.softmax(t, axis=3) * x
        n = tf.reduce_sum(input_tensor=n, axis=3)

        # handle variable-length day news sequences
        mask = tf.sequence_mask(day_len, maxlen=max_dlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=3)
        n *= mask

        # News-level attention
        u = self.u(n)
        d = tf.nn.softmax(u, axis=2) * n
        d = tf.reduce_sum(input_tensor=d, axis=2)

        # Sequence modeling
        h = self.bi_gru(d, training=training)

        # Temporal attention
        o = self.o(h)
        v = tf.nn.softmax(o, axis=2) * h
        v = tf.reduce_sum(input_tensor=v, axis=1)

        # Discriminative Network (MLP)
        v = self.fc0(v)
        v = self.dropout(v) if training else v
        v = self.fc1(v)
        v = self.dropout(v) if training else v
        return self.fc_out(v)

    def get_bi_gru(self, units):
        if tf.test.is_gpu_available() and not self.flags.no_gpu:
            return tf.keras.layers.Bidirectional(
                tf.compat.v1.keras.layers.CuDNNGRU(
                    units, return_sequences=True
                ), merge_mode='concat'
            )
        else:
            return tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units, return_sequences=True
                ), merge_mode='concat'
            )
 ```

### Train and Testing

```python

def train(model, optimizer, dataset, step_counter, ep, class_weights,
          log_interval=None):
    """Trains model on `dataset` using `optimizer`."""

    start = time.time()
    acc_train_epoch = 0.0
    loss_train_epoch = 0.0
    for step, ((days, day_lens, news_lens), labels) in enumerate(dataset):
        with tf.compat.v2.summary.record_if(tf.equal(0,step_counter % 50)):
               # 50, global_step=step_counter):

        #tf.compat.v2.summary.record_if() with the argument `lambda: tf.math.equal(0, global_step % n)
            # Record the operations used to compute the loss given the input,
            # so that the gradient of the loss with respect to the variables
            # can be computed.
            with tf.GradientTape() as tape:
                logits = model(days, day_lens, news_lens, training=True)
                loss_value = loss(logits, labels, class_weights)
                loss_train_epoch += loss_value
                accuracy = compute_accuracy(logits,labels)
                acc_train_epoch += accuracy
                tf.compat.v2.summary.scalar(name='loss', data=loss_value, step=tf.compat.v1.train.get_or_create_global_step())
                tf.compat.v2.summary.scalar(name='accuracy',
                                          data=compute_accuracy(logits, labels), step=tf.compat.v1.train.get_or_create_global_step())
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads, _ = clip_ops.clip_by_global_norm(grads,
                                                    model.flags.clip_norm)
            optimizer.apply_gradients(
                zip(grads, model.trainable_weights), global_step=step_counter)
            if log_interval and (step + 1) % log_interval == 0:
                rate = log_interval / (time.time() - start)
                print('Epoch #%d\tStep #%d\tLoss: %.6f (%.1f steps/sec)' % (
                    ep + 1, step, loss_value, rate))
                start = time.time()

            if ep == 0 and step == 0:
                print('#trainable_params', get_num_trainable_params(model))

    loss_train_epoch /= (step+1)
    acc_train_epoch /= (step+1)

    return loss_train_epoch, acc_train_epoch


def test(model, dataset, class_weights, show_classification_report=False,
         ds_name='Test'):
    start = time.time()
    """Perform an evaluation of `model` on the examples from `dataset`."""
    avg_loss = tf.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tf.metrics.Accuracy('accuracy', dtype=tf.float32)

    y_true = list()
    y_pred = list()
    acc_train_epoch = 0.0
    loss_train_epoch = 0.0
    for step, ((days, day_lens, news_lens), labels) in enumerate(dataset):        
        logits = model(days, day_lens, news_lens, training=False)
        avg_loss(loss(logits, labels, class_weights))
        loss_value = loss(logits,labels,class_weights)
        loss_train_epoch += loss_value
        pred = tf.argmax(input=logits, axis=1, output_type=tf.int64)
        accuracy(pred, tf.cast(labels, tf.int64))
        acc = compute_accuracy(logits,labels)
        acc_train_epoch += acc

        if show_classification_report:
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())
    end = time.time()

    acc_train_epoch /= (step + 1)
    loss_train_epoch /= (step + 1)
    print('%s set: Average loss: %.6f, Accuracy: %.3f%% (%.3f sec)' %
          (ds_name, avg_loss.result(), 100 * accuracy.result(), end - start))

    with tf.compat.v2.summary.record_if(True):
        tf.compat.v2.summary.scalar(name='loss', data=avg_loss.result(), step=tf.compat.v1.train.get_or_create_global_step())
        tf.compat.v2.summary.scalar(name='accuracy', data=accuracy.result(), step=tf.compat.v1.train.get_or_create_global_step())

    if show_classification_report:
        # print(classification_report(y_true, y_pred,
        #                             target_names=['PRESERVE', 'UP', 'DOWN']))
        print(classification_report(y_true, y_pred,
                                    target_names=['DOWN', 'UP']))  # StockNet

    return acc_train_epoch, loss_train_epoch #accuracy.result(), avg_loss.result()
```

### Results





## Directories
- src: source files;

## Running
