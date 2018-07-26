#!/usr/bin/python
# -*- coding:utf-8 -*-

__author__ = "Jiarong Xu"
__copyright__ = "Copyright 2017, The Anti-Plugin Project"
__version__ = "1.0.0"
__email__ = "xujr@zju.edu.cn"
__description__ = "基于action sequence的seq2seq模型训练"

import os
import json
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense


np.random.seed(7)

class seq2seq_train:
    def __init__(self, train_dataset_file, test_dataset_file, data_folder_list, model_file, logid_freq_file, logdesignid_freq_file, log_file, train_size, test_size):
        self.train_dataset_file = train_dataset_file       # 玩家序列文件训练集
        self.test_dataset_file = test_dataset_file         # 玩家序列文件验证集
        self.data_folder_list = data_folder_list           # 玩家序列文件子目录
        
        self.model_file = model_file                       # 模型存储文件
        self.log_file = log_file
#        self.logid_freq_file = logid_freq_file             # logid集合文件
#        self.logid_set = list()                            # logid集合
#        self.logid_int_to_vocab =dict()
#        self.logid_vocab_to_int =dict()
        
        self.logdesignid_freq_file = logdesignid_freq_file    # logdesignid集合文件
        self.logdesignid_set = list()                         # logdesignid集合
        self.logdesignid_int_to_vocab = dict()                # logdesignid词典
        self.logdesignid_vocab_to_int = dict()                # logdesignid词典
        self.logdesignid_vocab_size = 1000                    # logdesignid词典大小

        self.train_size = train_size                          # 训练集大小
        self.test_size = test_size                            # 验证集大小

        self.batch_size = 32
        self.epochs = 1
        self.learning_rate = 0.001
        self.maxlen = 1000                                    # Pad Sequence最大长度
        self.embedding_size_logdesignid = 32
        self.embedding_dropout_size = 0.5
        self.rnn_size = 128     # 64
        self.lstm_dropout_size = 0.5
        self.rnn_num_layers = 1
        
        self.display_step = 1  # Check training loss after every display_step batches
        self.saver_step = 100  # Check training loss after every saver_step batches

    def create_vocab(self):
        count = 0
        with open(self.logdesignid_freq_file, "r") as load_f:
            for line in load_f:
                self.logdesignid_set.append(line.strip())
                count += 1
                if count >= self.logdesignid_vocab_size:
                    break

        special_words = ['<PAD>', '<EOS>', '<GO>','<UNK>']
        self.logdesignid_int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(self.logdesignid_set))}
        self.logdesignid_vocab_to_int = {word: word_i for word_i, word in self.logdesignid_int_to_vocab.items()}  

    # Pad sentences with <PAD> so that each sentence of a batch has the same length
    def pad_sentence_batch(self, sentence_batch, pad_int, pad_eos, pad_go, codertype='encoder'):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        if max_sentence <= self.maxlen:
            to_return = [sentence+[pad_eos]+[pad_int]*(max_sentence - len(sentence)) if codertype=='encoder' 
                         else [pad_go]+sentence+[pad_int]*(max_sentence - len(sentence)) for sentence in sentence_batch]
        else:
            max_sentence = self.maxlen
            to_return = [sentence[:self.maxlen]+[pad_eos]+[pad_int]*(max_sentence-len(sentence)) if codertype=='encoder' 
                         else [pad_go]+sentence[:self.maxlen]+[pad_int]*(max_sentence - len(sentence)) for sentence in sentence_batch]
        return to_return
            
    def generator_batches(self,datatype):
        folder_list = self.data_folder_list
        dataset_file = self.train_dataset_file if datatype=='train' else self.test_dataset_file
        count = 0
        batch_player_logdesignid_data_enc = list()
        batch_player_logdesignid_data_dec = list()
        while(1):
            for folder in folder_list:
                filedir = dataset_file + folder
                for filenames in os.walk(filedir):
                    for filename in filenames[2]:
                        logdesignid_data_enc = list()
                        logdesignid_data_dec = list()
#                        logdesignid_data_dec.append(self.logdesignid_vocab_to_int['<GO>'])
                        player_event_dir = filedir + filename
                        with open(player_event_dir, "r") as load_f:
                            player_event = json.load(load_f)
                            if player_event==[]:
                                continue
                            for each_player_event in player_event:
#                                print(each_player_event['log_id'], each_player_event['design_id'])
#                                print(type(each_player_event['log_id']), type(each_player_event['design_id']))
                                letter_log = each_player_event['log_id'] if type(each_player_event['log_id'])==unicode else '<UNK>'
                                letter_design = each_player_event['design_id'] if type(each_player_event['design_id'])==unicode or type(each_player_event['design_id'])==int else '<UNK>'
#                                print(letter_log,letter_design)
                                letter = letter_log + '-' + str(letter_design) if letter_log!='<UNK>' and letter_design!='<UNK>' else '<UNK>'
#                                print(letter)
                                letter = self.logdesignid_vocab_to_int.get(letter, self.logdesignid_vocab_to_int['<UNK>'])
#                                print(letter)
#                                    letter = each_player_event['log_id'] if type(each_player_event['log_id'])==str else '<UNK>'
                                logdesignid_data_enc.append(letter)
                                logdesignid_data_dec.append(letter)
#                            logdesignid_data_enc.append(self.logdesignid_vocab_to_int['<EOS>'])
                        batch_player_logdesignid_data_enc.append(logdesignid_data_enc)
                        batch_player_logdesignid_data_dec.append(logdesignid_data_dec)
                        count += 1
                        if count == self.batch_size:
                            pad_enc_logdesignid_batch = np.array(self.pad_sentence_batch(batch_player_logdesignid_data_enc, 
                                                                                         self.logdesignid_vocab_to_int['<PAD>'], 
                                                                                         self.logdesignid_vocab_to_int['<EOS>'], 
                                                                                         self.logdesignid_vocab_to_int['<GO>'], codertype='encoder'))
                            pad_dec_logdesignid_batch = np.array(self.pad_sentence_batch(batch_player_logdesignid_data_dec, 
                                                                                         self.logdesignid_vocab_to_int['<PAD>'], 
                                                                                         self.logdesignid_vocab_to_int['<EOS>'], 
                                                                                         self.logdesignid_vocab_to_int['<GO>'], codertype='decoder'))
                            pad_target_logdesignid_batch = np.array(self.pad_sentence_batch(batch_player_logdesignid_data_enc, 
                                                                                            self.logdesignid_vocab_to_int['<PAD>'], 
                                                                                            self.logdesignid_vocab_to_int['<EOS>'], 
                                                                                            self.logdesignid_vocab_to_int['<GO>'], codertype='encoder'))
                                                                                            
                            # Need the lengths for the _lengths parameters
                            pad_source_lengths = []
                            for source in pad_enc_logdesignid_batch:
                                pad_source_lengths.append(len(source))
                            count = 0 # init again
                            batch_player_logdesignid_data_enc = list()
                            batch_player_logdesignid_data_dec = list()
                            yield pad_enc_logdesignid_batch, pad_dec_logdesignid_batch, pad_source_lengths, pad_target_logdesignid_batch

    def print_activations(self, t):
        print(t.op.name, t.get_shape().as_list())
    
    def get_model_inputs(self):
        with tf.name_scope('inputs'):
            input_data_logdesignid_enc = tf.placeholder(tf.int32, [None, None], name='input_logdesignid_enc')
            input_data_logdesignid_dec = tf.placeholder(tf.int32, [None, None], name='input_logdesignid_dec')
            target_logdesignid = tf.placeholder(tf.int32, [None, None], name='targets')
    
            lr = tf.placeholder(tf.float32, name='learning_rate')
        
            source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
            max_source_sequence_length = tf.reduce_max(source_sequence_length, name='max_source_len')
            
        return input_data_logdesignid_enc, input_data_logdesignid_dec,target_logdesignid, lr, source_sequence_length, max_source_sequence_length
                
 
    def encoding_layer(self, enc_embed_input, rnn_size, num_layers,
                       source_sequence_length, source_vocab_size):  
        
        with tf.variable_scope("encode"):  
        # RNN cell
            def make_cell(rnn_size):
                enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return enc_cell
            enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])       
            enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
            print("[encoding_layer] encoder output shape:")
            self.print_activations(enc_output)
#        self.print_activations(enc_state)
        return enc_output, enc_state     

    def decoding_layer(self, target_letter_to_int, num_layers, rnn_size,
                       target_sequence_length, max_target_sequence_length, enc_state, dec_embed_input):
        # 1. Decoder Embedding
        target_vocab_size = len(target_letter_to_int)
        print("[decoding_layer] target vocab size:" + str(target_vocab_size))
#        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
#        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
        with tf.variable_scope("decode"):  
        # 2. Construct the decoder cell
            def make_cell(rnn_size):
                dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return dec_cell    
            dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
             
            # 3. Dense layer to translate the decoder's output at each time 
            # step into a choice from the target vocabulary
            output_layer = Dense(target_vocab_size,
                                 kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
#        print("[decoding_layer] Dense output_layer", tf.shape(output_layer))
#        self.print_activations(output_layer)
    
        # 4. Set up a training decoder and an inference decoder
        # Training Decoder  
            # Helper for the training process. Used by BasicDecoder to read inputs.
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                                sequence_length=target_sequence_length,
                                                                time_major=False)
            # Basic decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                               training_helper,
                                                               enc_state,
                                                               output_layer)            
            # Perform dynamic decoding using the decoder
            training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)[0]  
#        print("[decoding_layer] training_decoder_output")
#        self.print_activations(training_decoder_output)
        return training_decoder_output


    def seq2seq_model(self, input_data_enc, input_data_dec, targets, lr, target_sequence_length, 
                      max_target_sequence_length, source_sequence_length,
                      source_vocab_size, target_vocab_size,
                      lstm_size, num_layers):
        
        print('[seq2seq_model] enc_input',input_data_enc.get_shape())
        # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
        _, enc_state = self.encoding_layer(input_data_enc, 
                                      lstm_size, 
                                      num_layers, 
                                      source_sequence_length,
                                      source_vocab_size)
        
        
        # Prepare the target sequences we'll feed to the decoder in training mode
#        dec_input = self.process_decoder_input(targets)
        print('[seq2seq_model] dec_input',input_data_dec.get_shape())
        
        # Pass encoder state and decoder inputs to the decoders
        training_decoder_output = self.decoding_layer(self.logdesignid_vocab_to_int,
                                                      num_layers, 
                                                      lstm_size,
                                                      target_sequence_length,
                                                      max_target_sequence_length,
                                                      enc_state, 
                                                      input_data_dec)

        return training_decoder_output, enc_state

    def model_train(self):
        # 1. Build the graph
        train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training
        with train_graph.as_default():
            # Load the model inputs
            input_data_logdesignid_enc, input_data_logdesignid_dec, target_logdesignid, lr, source_sequence_length, max_source_sequence_length = self.get_model_inputs()
            
            print('EMBEDDING! vocal_size= ' + str(len(self.logdesignid_int_to_vocab)) + ' ,embed_dim= ' + str(self.embedding_size_logdesignid))
            enc_embed_input_logdesignid = tf.contrib.layers.embed_sequence(input_data_logdesignid_enc, vocab_size = len(self.logdesignid_int_to_vocab), embed_dim = self.embedding_size_logdesignid)
            print("[model_train] enc_embed_input_logdesignid:")
            self.print_activations(enc_embed_input_logdesignid)
            dec_embed_input_logdesignid = tf.contrib.layers.embed_sequence(input_data_logdesignid_dec, vocab_size = len(self.logdesignid_int_to_vocab), embed_dim = self.embedding_size_logdesignid)
            print("[model_train] dec_embed_input_logdesignid:")
            self.print_activations(dec_embed_input_logdesignid)
            
            input_data_enc = enc_embed_input_logdesignid
            input_data_dec = dec_embed_input_logdesignid
#            targets = input_data_enc #autoencoder: target equals to input
#            targets = self.player_logid_test
            
            with tf.name_scope('seq2seq'):
            # Create the training and inference logits
                training_decoder_output, enc_state = self.seq2seq_model(input_data_enc,
                                                             input_data_dec,
                                                             target_logdesignid,
                                                             lr,
                                                             source_sequence_length,
                                                             max_source_sequence_length,
                                                             source_sequence_length,
                                                             len(self.logdesignid_int_to_vocab),
                                                             len(self.logdesignid_int_to_vocab),
                                                             self.rnn_size,
                                                             self.rnn_num_layers)
            # Create tensors for the training logits and inference logits
            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
#            print("[model train] training_logits:")
#            self.print_activations(training_logits)

            # Create the weights for sequence_loss
            masks = tf.sequence_mask(source_sequence_length, max_source_sequence_length, dtype=tf.float32, name='masks')
#            print("[model train] masks:")
#            self.print_activations(masks)

            with tf.name_scope("optimization"):
                # Loss function
                print('[model_train] training_logits:',training_logits.get_shape())
                print('[model_train] targets',target_logdesignid.get_shape())
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    target_logdesignid,
                    masks)
                tf.summary.scalar('loss', cost)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(lr)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                    grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

        
        # 2.Start train
        checkpoint = self.model_file + "best_model.ckpt"
        
        with tf.Session(graph=train_graph) as sess: 
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.log_file + 'train',sess.graph)
            test_writer = tf.summary.FileWriter(self.log_file + 'test')
            sess.run(tf.global_variables_initializer())

            max_batchsize = self.train_size // self.batch_size
            # for epoch_i in range(1, self.epochs + 1):
            epoch_i = 1
            test_generator = self.generator_batches(datatype='test')
            for batch_i, (pad_enc_logdesignid_batch, pad_dec_logdesignid_batch, sources_lengths, train_targets_batch) in enumerate(
                    self.generator_batches(datatype='train')):

#                    print('train_targets_batch',train_targets_batch)
                if (batch_i % max_batchsize)+1 == max_batchsize:
                    epoch_i += 1
                    if epoch_i >= self.epochs:
                        break
                # Training step
                with tf.name_scope('loss'):
#                    try:
#                    print('train',pad_enc_logdesignid_batch,pad_dec_logdesignid_batch,train_targets_batch)
                    summary, _, loss = sess.run(
                        [merged, train_op, cost],
                        {input_data_logdesignid_enc: pad_enc_logdesignid_batch,
                         input_data_logdesignid_dec: pad_dec_logdesignid_batch,
                         target_logdesignid: train_targets_batch,
                         lr: self.learning_rate,
                         source_sequence_length: sources_lengths,
                         })
#                    except:
                    
                    train_writer.add_summary(summary, batch_i)

                # Debug message updating us on the status of the training
                if batch_i % self.display_step == 0:
                    
                    (pad_enc_valid_logdesignid_batch, pad_dec_valid_logdesignid_batch, valid_sources_lengths,
                     valid_targets_batch) = next(test_generator)
                    # Calculate validation cost
#                    print('test',pad_enc_valid_logdesignid_batch,pad_dec_valid_logdesignid_batch,valid_targets_batch)
                    summary, validation_loss = sess.run(
                        [merged, cost],
                        {input_data_logdesignid_enc: pad_enc_valid_logdesignid_batch,
                         input_data_logdesignid_dec: pad_dec_valid_logdesignid_batch,
                         target_logdesignid: valid_targets_batch,
                         lr: self.learning_rate,
                         source_sequence_length: valid_sources_lengths,
                         })
                    test_writer.add_summary(summary, batch_i)

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  self.epochs,
                                  (batch_i % max_batchsize)+1,
                                  max_batchsize,
                                  loss,
                                  validation_loss))

#                if epoch_i % self.saver_step == 0 and ((batch_i % max_batchsize)+1) % max_batchsize == 0:
                if ((batch_i % max_batchsize)+1) % self.saver_step == 0:
                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(os.getcwd(), self.model_file + "epoch"+str(epoch_i)+"batch"+str((batch_i % max_batchsize)+1)+".ckpt"))
                 

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
            print('Model Trained and Saved')

    def run(self):
        self.create_vocab()                                     # 加载id集合并创建int-vocab词汇表
        print('--------------CREATE VOCAB FINISH!--------------')
        self.model_train()                                      # 模型训练

def countsize(mainpath, folder_list):
    count = 0
    for folder in folder_list:
        for fn in os.listdir(mainpath+folder):
            count = count+1
    return count

def dir_check(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':

    start = time.time()

    # 参数1 数据集名称
#    dataset_file = 'data/' + sys.argv[1]
    train_dataset_file = 'data/datatrain/'
    test_dataset_file = 'data/datatest/'
    data_folder_list = ['new_create_grade_30/']

    logid_freq_file = ''
    logdesignid_freq_file = 'count/logdesignid_all'
    
    model_file = 'model/logdesignid_128_13000_5000/'
    dir_check(model_file)
    log_file = 'log/logdesignid_128_13000_5000/'
    dir_check(log_file)
    
    train_size = countsize(train_dataset_file, data_folder_list)
    test_size = countsize(test_dataset_file, data_folder_list)
    print('train_size',train_size)
    print('test_size',test_size)
      
    seq2seq_train_ins =seq2seq_train(train_dataset_file, test_dataset_file, data_folder_list, model_file, logid_freq_file, logdesignid_freq_file, log_file, train_size, test_size)
    seq2seq_train_ins.run()

    stop = time.time()
    print("模型训练时间: " + str(stop - start) + "秒")             # 统计模型训练时间
