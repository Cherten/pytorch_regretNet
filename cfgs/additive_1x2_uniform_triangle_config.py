import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.dir_name = os.path.join("experiments", "additive_1x2_uniform_triangle")
__C.num_agents = 1
__C.num_items = 2
__C.distribution_type = "uniform_triangle"
__C.agent_type = "additive"
__C.save_data = False
__C.net = edict()
__C.net.init = "gu"
__C.net.activation = "tanh"
__C.net.num_a_layers = 3
__C.net.num_p_layers = 3
__C.net.num_p_hidden_units = 50
__C.net.num_a_hidden_units = 50
__C.train = edict()
__C.train.seed = 42
__C.train.restore_iter = 0
__C.train.max_iter = 2000
__C.train.learning_rate = 1e-3
__C.train.wd = None
__C.train.data = "fixed"
__C.train.num_batches = 5000
__C.train.batch_size = 64
__C.train.adv_reuse = True
__C.train.num_misreports = 1
__C.train.gd_iter = 25
__C.train.gd_lr = 0.1
__C.train.update_rate = 1.0
__C.train.w_rgt_init_val = 5.0
__C.train.update_frequency = 100
__C.train.up_op_add = 100.0
__C.train.up_op_frequency = 10000
__C.train.max_to_keep = 20
__C.train.save_iter = 500
__C.train.print_iter = 100
__C.val = edict()
__C.val.gd_iter = 2000
__C.val.gd_lr = 0.1
__C.val.num_batches = 20
__C.val.print_iter = 500
__C.val.data = "fixed"
__C.test = edict()
__C.test.seed = 100
__C.test.restore_iter = 400000
__C.test.num_misreports = 1000
__C.test.gd_iter = 2000
__C.test.gd_lr = 0.1
__C.test.data = "online"
__C.test.num_batches = 100
__C.test.batch_size = 100
__C.test.save_output = False
__C.val.batch_size = __C.train.batch_size
__C.val.num_misreports = __C.train.num_misreports
__C.train.num_instances = __C.train.num_batches * __C.train.batch_size
__C.val.num_instances = __C.val.num_batches * __C.val.batch_size
__C.test.num_instances = __C.test.num_batches * __C.test.batch_size
