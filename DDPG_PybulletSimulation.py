import Deepdeterministic
import tensorflow as tf
import time
import keras
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto()
#config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(video_name = None):
    s_size = 19
    with tf.device('/cpu'):
        with tf.Session() as session:
            actor = Deepdeterministic.Actor(session, a_bound=1., s_size=s_size, a_size=2)
            critic = Deepdeterministic.Critic(session, s_size=s_size, a_size=2)

            saver = tf.train.Saver() 
            save_path = saver.restore(session, "data/Nabeel/tf/DDPG.ckpt")
            print("model restored")

            for i in range(3):
                s = time.time()
                env = Deepdeterministic.Environment(render=True, sigma=0.02, down=1.0, get_image=False)

                if video_name:
                    env.record("data/Nabeel/Video/" + video_name + ".mp4")

                step = env.replay(actor.policy_one)

                print("unicycle lasted {} steps and {:2f} seconds.".format(step, step / 30))
                print("time = {}".format(time.time() - s))
                env.close()

    end = time.time()

if __name__ == '__main__':
    main(video_name="DDPG")



