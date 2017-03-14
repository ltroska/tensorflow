# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"],
                                    "hpx_root": ["localhost:2222"]})

server = tf.train.Server(cluster, job_name="local", task_index=task_number, protocol="hpx")

print("Starting server #{}".format(task_number))

server.start()
server.join()
