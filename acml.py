import tensorflow as tf
from tensorflow.keras.layers import Layer

class AdaptiveChannelMixingLayer(Layer):
    def __init__(self, channels, **kwargs):
        super(AdaptiveChannelMixingLayer, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        
        self.mix_weights = self.add_weight(
            shape=(self.channels, self.channels),
            initializer='he_normal',
            trainable=True,
            name='mix_weights'
        )
        
        self.control_weights = self.add_weight(
            shape=(self.channels,),
            initializer='ones',
            trainable=True,
            name='control_weights'
        )
        super(AdaptiveChannelMixingLayer, self).build(input_shape)

    def call(self, inputs):
       
        mixed_signals = tf.tensordot(inputs, self.mix_weights, axes=[[2], [0]])
        
        corrected_signals = inputs + mixed_signals * self.control_weights
        # corrected_signals = BatchNormalization(epsilon=1e-05, momentum=0.9,axis=-1)(corrected_signals)
        return corrected_signals

    def get_config(self):
        config = super(AdaptiveChannelMixingLayer, self).get_config()
        config.update({"channels": self.channels})
        return config

def build_model_mixchannel(class_model,channels,samples):
    eeg_input = Input(shape=(1,channels,samples))
    input2 = Permute((1,3,2))(eeg_input)
    input2 = K.squeeze(input2,axis=1)
   
    adp_conf =  AdaptiveChannelMixingLayer(channels=channels)(input2)
    adp_conf = Permute((2,1))(adp_conf)
    adp_out = tf.expand_dims(adp_conf,axis=1)

   
    align_model = Model(eeg_input,adp_out)
    align_layer = align_model(eeg_input)

    # class_model.summary()
    class_pred = class_model(align_layer) #suqueenze the ACML with other model


    train_model = Model(eeg_input,class_pred)
    
    adm = adam_v2.Adam(learning_rate=9e-4, beta_1=0.9, beta_2=0.999,amsgrad=False,epsilon=1e-08, decay=0.0, clipnorm=1.0)
    train_model.compile(optimizer=adm, loss='categorical_crossentropy',metrics = ['accuracy'])
    return train_model
