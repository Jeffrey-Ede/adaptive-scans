import tensorflow as tf

import itertools

FLAGS = tf.flags.FLAGS

def auto_name(name):
    """Append number to variable name to make it unique.
    
    Inputs:
        name: Start of variable name.

    Returns:
        Full variable name with number afterwards to make it unique.
    """

    scope = tf.contrib.framework.get_name_scope()
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    names = [v.name for v in vars]
    
    #Increment variable number until unused name is found
    for i in itertools.count():
        short_name = name + "_" + str(i)
        sep = "/" if scope != "" else ""
        full_name = scope + sep + short_name
        if not full_name in [n[:len(full_name)] for n in names]:
            return short_name

def alrc(
    loss, 
    num_stddev=3, 
    decay=0.999, 
    mu1_start=1, 
    mu2_start=2**2, 
    in_place_updates=False
    ):
    """Adaptive learning rate clipping (ALRC) of outlier losses.
    
    Inputs:
        loss: Loss function to limit outlier losses of.
        num_stddev: Number of standard deviation above loss mean to limit it
        to.
        decay: Decay rate for exponential moving averages used to track the first
        two raw moments of the loss.
        mu1_start: Initial estimate for the first raw moment of the loss.
        mu2_start: Initial estimate for the second raw moment of the loss.
        in_place_updates: If False, add control dependencies for moment tracking
        to tf.GraphKeys.UPDATE_OPS. This allows the control dependencies to be
        executed in parallel with other dependencies later.
    Return:
        Loss function with control dependencies for ALRC.
    """

    #Varables to track first two raw moments of the loss
    mu = tf.get_variable(
        auto_name("mu1"), 
        initializer=tf.constant(mu1_start, shape=[1], dtype=tf.float32))
    mu2 = tf.get_variable(
        auto_name("mu2"), 
        initializer=tf.constant(mu2_start, shape=[1], dtype=tf.float32))

    #Use capped loss for moment updates to limit the effect of outlier losses on the threshold
    sigma = tf.sqrt(mu2 - mu**2+1.e-8)
    loss = tf.where(loss < mu+num_stddev*sigma, 
                   loss, 
                   loss/tf.stop_gradient(loss/(mu+num_stddev*sigma)))

    #Update moment moving averages
    mean_loss = tf.reduce_mean(loss, axis=0, keepdims=True)
    mean_loss2 = tf.reduce_mean(loss**2, axis=0, keepdims=True)
    update_ops = [mu.assign(decay*mu+(1-decay)*mean_loss), 
                  mu2.assign(decay*mu2+(1-decay)*mean_loss2)]
    if in_place_updates:
        with tf.control_dependencies(update_ops):
            loss = tf.identity(loss)
    else:
        #Control dependencies that can be executed in parallel with other update
        #ops. Often, these dependencies are added to train ops e.g. alongside
        #batch normalization update ops.
        for update_op in update_ops:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        
    return loss


if __name__ == "__main__":
    
    pass