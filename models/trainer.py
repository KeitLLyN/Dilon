import tensorflow as tf
import numpy as np
import timeit
import utils.service as service


class Trainer:

    def __init__(self, batch_size, checkpoints_path, trainer):
        self.batch_size = batch_size
        self.checkpoints = checkpoints_path
        self.path_to_graph = checkpoints_path + 'seq2seq'
        self.dropout = trainer

    def train(self, model, train_data, train_size, num_steps, num_epochs, min_loss=0.3):
        """
        Trains a given model architecture with given train data.
        """
        tf.compat.v1.set_random_seed(1234)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            total_loss = []
            timings = []
            steps_per_epoch = int(train_size / self.batch_size)
            num_epoch = 1

            for step in range(1, num_steps):
                beg_t = timeit.default_timer()
                x, y = train_data.next()
                seq_len = np.max(y)

                # For anomaly detection problem we reconstruct input data, so
                # targets and inputs are identical.
                feed_dict = {
                    model.inputs: x,
                    model.targets: x,
                    model.lengths: y,
                    model.dropout: self.dropout,
                    model.batch_size: self.batch_size,
                    model.max_seq_len: seq_len}

                fetches = [model.loss, model.decoder_outputs, model.train_optimizer]
                step_loss, _, _ = sess.run(fetches, feed_dict)

                total_loss.append(step_loss)
                timings.append(timeit.default_timer() - beg_t)

                if step % steps_per_epoch == 0:
                    num_epoch += 1

                if step % 200 == 0 or step == 1:
                    service.print_progress(
                        int(step / 200),
                        num_epoch,
                        np.mean(total_loss),
                        np.mean(step_loss),
                        np.sum(timings))
                    timings = []
                if step == 1:
                    _ = tf.compat.v1.train.export_meta_graph(filename=self.path_to_graph + '.meta')

                if np.mean(total_loss) < min_loss or num_epoch > num_epochs:
                    model.saver.save(sess, self.path_to_graph, global_step=step)
                    print("Training is finished.")
                    break
