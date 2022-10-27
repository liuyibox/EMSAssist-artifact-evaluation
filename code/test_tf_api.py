import tensorflow as tf
import os

def test_onehot():

    # some sparse features, from raw maker_id to index
    maker_id_list = [1, 3, 9, 14, 2, 3]
    one_hot_enc = tf.one_hot(indices=maker_id_list, depth=16)
    
    #with tf.Session() as sess:
    feature_list = tf.math.reduce_max(one_hot_enc, axis=0)
    print(feature_list)
    print(type(feature_list.numpy()))
    print(feature_list.numpy())

    inputs1 = tf.keras.layers.InputLayer(input_shape=(100,), name="ANNInput")
    print(inputs1.input_shape)
    inputs2 = tf.keras.Input(shape=(100,), name="ANNInput")
    print(inputs2.shape)


    layer2 = tf.keras.layers.Dense(
            100,
            name="layer2",
            activation="softmax",
            dtype=tf.float32)(inputs2)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
    test_onehot()
