# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

#syntax: same as regular tensorflow clusterspec, except there is an additional job called "hpx_root"
#        which is used to determine which server is AGAS
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223", "localhost:2224"],
                                    "hpx_root": ["localhost:2222"]})

#syntax: same as regular tensorflow, except setting the protocol to either
#   "hpx", for execution using the hpx distributed services, using the HPX executor locally
#   "hpx.tf", for execution using the hpx distributed services, using the TF executor locally
server = tf.train.Server(cluster, job_name="local", task_index=task_number, protocol="hpx")

print("Starting server #{}".format(task_number))

server.start()
server.join()
