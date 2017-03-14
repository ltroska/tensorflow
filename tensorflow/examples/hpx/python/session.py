import tensorflow as tf


cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"]})

x = tf.constant(2)




with tf.device("/job:local/task:1"):
    y2 = x - 66

#with tf.device("/job:local/task:2"):
#    y3 = y2 * 2

with tf.device("/job:local/task:0"):
    y1 = x + 300
    y =  y1 + y2# + y3


with tf.Session("hpx://localhost:2222") as sess:

    result = sess.run(y)
    print(result)
    
