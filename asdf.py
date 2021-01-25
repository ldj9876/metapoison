import keras
import numpy as np
import tensorflow as tf

class args:
    num_class = 10
    num_batch = 40
    batch_size = 125
    num_valid = 1000

    n_craftstep = 60
    craft_rate = 200
    craft_rate_drop_period = 20

    model_replay = 8
    model_name = 'resnet50'
    learning_rate = 0.1
    n_adapt = 2 # in algorithm unrolled step K

def get_resnet(model_name,input_shape, num_class=10):
    if model_name == 'resnet50':
        model = keras.applications.ResNet50(include_top=True, weights=None,classes=num_class,input_shape=input_shape)
    elif model_name == 'resnet101':
        model = keras.applications.ResNet50(include_top=True, weights=None,classes=num_class,input_shape=input_shape)
    elif model_name == 'resnet152':
        model = keras.applications.ResNet50(include_top=True, weights=None,classes=num_class,input_shape=input_shape)

    else:
        raise ValueError(f'Invalid ResNet model chosen: {model_name}.')

    return model

def model_initalize(model):
    pass
def get_batch():
    pass
def adv_loss():
    pass

n_craftstep = args.n_craftstep
craft_rate = args.craft_rate
craft_rate_drop_period = args.craft_rate_drop_period
model_replay = args.model_replay
model_name = args.model_name
learning_rate = args.learning_rate
batch_size = args.batch_size

train_loss = keras.losses.categorical_crossentropy
model = get_resnet(model_name,(32,32,3))
mod = model.get_weights()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                          decay_steps=50,
                                                          decay_rate=10)
train_optimizer = keras.optimizers.SGD(learning_rate = lr_schedule)


for craftstep in range(1,n_craftstep+1):
    if craftstep % craft_rate_drop_period == 0:
        craft_rate = craft_rate / 10
    adv_grads_hist = []
    for replay in range(model_replay):
        model_initalize(model)


        prev_weight = model.get_weights()
        for _ in range(args.n_adapt):
            x_batch, y_batch = get_batch()
            with tf.GradientTape as tape:
                logits = model(x_batch, training = True)
                loss_value = train_loss(y_batch,logits)
            grads = tape.gradient(loss_value,model.trainable_weights)

            train_optimizer.apply_gradients(zip(grads,model.trainable_weights))

        with tf.GradientTape as tape:
            target = poison_model(target_ids)
            adv_logit = model(targets, training = False)
            loss = adv_loss(adv_label, adv_logit)

        grads = tape.gradient(loss_value, poison_model.trainable_weights)
        adv_grad_hist.append(grads)

        model.set_weights(prev_weight)

        for x_batch, y_batch in train_set:
            with tf.GradientTape as tape:
                logits = model(x_batch,training = True)
                loss_value = train_loss(y_batch,logits)

            grads = tape.gradient(loss_value,model.trainable_weights)

            train_optimizer.apply_gradients(zip(grads,model.trainable_weights))

    adv_grad = tf.reduce_mean(adv_grads_hist,axis=0)
    poison_optimizer.apply_graidents(zip(adv_grad,poison_model.trainable_weights))










