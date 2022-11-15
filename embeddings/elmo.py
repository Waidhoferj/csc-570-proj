import tensorflow_hub as hub
import tensorflow.compat.v1 as tf


def get_elmo_embeddings(sent_list):
    tf.disable_eager_execution()
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    embeddings = elmo(sent_list)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    return embeddings.eval(session=sess)


if __name__ == "__main__":
    sent_list = ["I love hot dogs", "My dogs are so cute"]
    elmo_embeddings = get_elmo_embeddings(sent_list)
