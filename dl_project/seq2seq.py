import tensorflow as tf


class seq2seq:

	def __init__(self, vs, ies, hu):
		self.vocab_size = vs + 2
		self.input_embedding_size = ies
		self.encoder_hidden_units = hu
		self.decoder_hidden_units = hu


	def build(self):
		self.encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
		self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')

		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)
		self.encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
		self.decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

		encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)
		self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs_embedded, dtype=tf.float32, time_major=True)

		decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_units)
		self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, self.decoder_inputs_embedded, initial_state=self.encoder_final_state,
											dtype=tf.float32, time_major=True, scope='plain_decoder')

		self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
		self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

