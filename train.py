# GUI dependencies
from gooey import Gooey, GooeyParser

# other dependencies
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model


@Gooey(required_cols=2,
       program_name="Training - End to End Self Driving",
       progress_regex=r"^Epoch: (?P<current>\d+)/(?P<total>\d+).*$",
       progress_expr="current / total * 100",
       timing_options={
           'show_time_remaining': True,
           'hide_time_remaining_on_complete': True
       }
       )
def parse_args():
    prog_description = 'Configure various hyperparameters for training the deep learning model. '
    parser = GooeyParser(description=prog_description)

    sub_parsers = parser.add_subparsers(help='commands', dest='command')

    train_parser = sub_parsers.add_parser('train')

    train_parser.add_argument('--epochs', default=30, type=int)
    train_parser.add_argument('--batch-size', default=100, type=int)
    train_parser.add_argument('--l2-norm-const', default=0.001, type=float)

    return parser.parse_args()


def train(epochs=30, batch_size=100, l2_norm_const=0.001):
    LOGDIR = './save'
    sess = tf.InteractiveSession()

    train_vars = tf.trainable_variables()

    loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n(
        [tf.nn.l2_loss(v) for v in train_vars]) * l2_norm_const
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.global_variables_initializer())

    # create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

    # op to write logs to Tensorboard
    logs_path = './logs'
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # train over the dataset about 30 times
    for epoch in range(epochs):
        for i in range(int(driving_data.num_images / batch_size)):
            xs, ys = driving_data.LoadTrainBatch(batch_size)
            train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
            if i % 10 == 0:
                xs, ys = driving_data.LoadValBatch(batch_size)
                loss_value = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
                print("Epoch: %d/%d, Step: %d, Loss: %g" % (epoch, epochs, epoch * batch_size + i, loss_value))

            # write logs at every iteration
            summary = merged_summary_op.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
            summary_writer.add_summary(summary, epoch * driving_data.num_images / batch_size + i)

            if i % batch_size == 0:
                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)
                checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
                filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)

    print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")


if __name__ == '__main__':
    conf = parse_args()
    print(conf)
    train(epochs=conf.epochs, batch_size=conf.batch_size, l2_norm_const=conf.l2_norm_const)
