import tensorflow as tf

#syntax: same as regular tensorflow clusterspec, except there is an additional job called "hpx_root"
#        which is used to determine which server is AGAS
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"], "hpx_root": ["localhost:2222"]})

x = tf.constant(2)


with tf.device("/job:local/task:1"):
    y2 = x - 66

with tf.device("/job:local/task:0"):
    y1 = x + 300
    y =  y1 + y2


#syntax: tf.Session("hpx://<target>|<agas>|<host>")
#where:
#   <target> is the node to run the session on
#   <agas> is the node that runs AGAS (defined by hpx_root above)
#   <host> is optional and defines the ip and port for this session
with tf.Session("hpx://localhost:2222") as sess:
    result = sess.run(y)
    print(result)
    
