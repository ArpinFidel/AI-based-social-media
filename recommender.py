import numpy as np
import pandas as pd
import tensorflow as tf
import pymysql
import math
import datetime
import time

from sklearn import preprocessing
from sklearn.metrics import precision_score
from random import shuffle

display_step = 10
learning_rate = 0.3
epoch_cnt = 50

k = 10

hidden1_cnt = 10
hidden2_cnt = 5

class Connect(object):
	def __enter__(self):
		self.conn = pymysql.connect(host='localhost', user='root', db='chat')
		return self.conn, self.conn.cursor()
	def __exit__(self, type, value, traceback):
		try: self.conn.commit()
		finally: self.conn.close()

if True:
# while True:
	
# 	now = datetime.datetime.now()
# 	if now.minute != 0:
# 		time.sleep(((60-now.minute)*60 - now.second)*bool(now.minute))
# 		continue
	
	with Connect() as (conn, cursor):
		sql = "SELECT `from_user`, `to_user`, `rate` FROM rating"
		cursor.execute(sql)
		result = cursor.fetchall()
			
	df = pd.DataFrame(list(result), columns=['user', 'item', 'rating'])
	
	item_cnt = df.item.nunique()
	user_cnt = df.user.nunique()

	batch_sz = min(max(1, min(item_cnt, user_cnt)/2), 100)
	
	mat = df.pivot(index='user', columns='item', values='rating')
	mat.fillna(0, inplace=True)

	users = mat.index.tolist()
	items = mat.columns.tolist()

	mat = mat.values

	input_cnt = item_cnt

	X = tf.placeholder(tf.float64, [None, input_cnt])

	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([input_cnt, hidden1_cnt], dtype=tf.float64)),
		'encoder_h2': tf.Variable(tf.random_normal([hidden1_cnt, hidden2_cnt], dtype=tf.float64)),
		'decoder_h1': tf.Variable(tf.random_normal([hidden2_cnt, hidden1_cnt], dtype=tf.float64)),
		'decoder_h2': tf.Variable(tf.random_normal([hidden1_cnt, input_cnt], dtype=tf.float64)),
	}

	biases = {
		'encoder_b1': tf.Variable(tf.random_normal([hidden1_cnt], dtype=tf.float64)),
		'encoder_b2': tf.Variable(tf.random_normal([hidden2_cnt], dtype=tf.float64)),
		'decoder_b1': tf.Variable(tf.random_normal([hidden1_cnt], dtype=tf.float64)),
		'decoder_b2': tf.Variable(tf.random_normal([input_cnt], dtype=tf.float64)),
	}

	def encoder(x):
		l_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
		l_2 = tf.nn.sigmoid(tf.add(tf.matmul(l_1, weights['encoder_h2']), biases['encoder_b2']))
		return l_2

	def decoder(x):
		l_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
		l_2 = tf.nn.sigmoid(tf.add(tf.matmul(l_1, weights['decoder_h2']), biases['decoder_b2']))
		return l_2

	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	y_pred = decoder_op
	y_true = X

	loss = tf.losses.mean_squared_error(y_true, y_pred)
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

	preds = pd.DataFrame()

	global_init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()

	with tf.Session() as session:
		session.run(global_init)
		session.run(local_init)

		batch_cnt = math.ceil(mat.shape[0]/batch_sz)
		mat = np.array_split(mat, batch_cnt)

		for i in range(epoch_cnt):
			for batch in mat:
				session.run([optimizer, loss], feed_dict={X: batch})[1]

		mat = np.concatenate(mat, axis=0)

		preds = preds.append(pd.DataFrame(session.run(decoder_op, feed_dict={X: mat})))
		preds = preds.stack().reset_index(name='rating')
		preds.columns = ['user', 'item', 'rating']
		
		preds['user'] = preds['user'].map(lambda value: users[value])
		preds['item'] = preds['item'].map(lambda value: items[value])

		keys = ['user', 'item']
		
		tmp1 = preds.set_index(keys).index
		tmp2 = df[df['rating'] != 0].set_index(keys).index

		recs = preds[~tmp1.isin(tmp2)]
		recs = recs[recs['user'] != recs['item']]
		
		recs = recs.sort_values(['user', 'rating'], ascending=[True, False])
		recs = recs.groupby('user').head(k)
		
		with Connect() as (conn, cursor):
			sql = "DROP TABLE IF EXISTS prediction"
			cursor.execute(sql)
		
		with Connect() as (conn, cursor):
			sql = "CREATE TABLE prediction (`from_user` int, `to_user` int, `rate` float)"
			cursor.execute(sql) 
		
		for row in recs.iterrows():
			with Connect() as (conn, cursor):
				sql = "INSERT INTO prediction (`from_user`, `to_user`, `rate`) VALUES (%s, %s, %s)"
				cursor.execute(sql, (int(row[1]['user']), int(row[1]['item']), float(row[1]['rating'])))
				