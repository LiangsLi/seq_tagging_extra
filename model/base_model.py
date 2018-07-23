import os
import tensorflow as tf


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""
    
    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None
    
    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer
            初始化层权重
        """
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)
    
    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch
            定义训练步骤
        Args:
            lr_method: (string) sgd method, for example "adam" 优化器名字
            lr: (tf.placeholder) tf.float32, learning rate  学习率
            loss: (tensor) tf.float32 loss to minimize  **损失**
            clip: (python float) clipping of gradient. If < 0, no clipping 梯度裁剪
            方法执行后，得到self.train_op(反向传播的训练过程)
        """
        _lr_m = lr_method.lower()  # lower to make sure
        
        with tf.variable_scope("train_step"):
            #  设置优化器：
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            
            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                # 执行梯度裁剪：
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)
    
    def initialize_session(self):
        """Defines self.sess and initialize the variables
        创建model所属的sess，同时执行全局初始化，同时设置self.saver
        """
        self.logger.info("Initializing tf session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # session = tf.Session(config=config, ...)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    
    def restore_session(self, dir_model):
        """Reload weights into session
        加载预先保存的model？？
        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)
    
    def save_session(self):
        """Saves session = weights
        保存模型？？
        """
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)
    
    def close_session(self):
        """Closes the session
        关闭session
        """
        self.sess.close()
    
    def add_summary(self):
        """Defines variables for Tensorboard
        保存tensorboard所需的信息
        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                                                 self.sess.graph)
    
    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard
        
        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                                                           self.config.nepochs))
            # run_epoch执行一个epoch 的训练op（在子类中实现）
            score = self.run_epoch(train, dev, epoch)
            # 变化学习率
            self.config.lr *= self.config.lr_decay  # decay learning rate
            
            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without " \
                                     "improvement".format(nepoch_no_imprv))
                    break
    
    def evaluate(self, test, eval_file=None):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        # run_evaluate执行一次预测（在子类中实现）
        metrics = self.run_evaluate(test, eval_file)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)
