
"""
Determine the number of training steps.
"""
function get_train_steps(num_examples)
    return (num_examples * train_epochs ÷ train_batch_size + 1)
end

"""
Build learning rate schedule


global_step refers to the number of batches seen by the graph. 
Every time a batch is provided, the weights are updated in the direction that minimizes the loss. 
global_step just keeps track of the number of batches seen so far.
When it is passed in the minimize() argument list, the variable is increased by one. 
Have a look at optimizer.minimize().
You can get the global_step value using tf.train.global_step(). 
Also handy are the utility methods tf.train.get_global_step or tf.train.get_or_create_global_step.


"""
function learning_rate_schedule(base_learning_rate, num_examples)
    #global_step = tf.train.get_or_create_global_step()
    warmup_steps = round(Int, warmup_epochs * num_examples ÷ train_batch_size)
    ## WHY WE DIVIDE BY 256.
    scaled_lr = base_learning_rate * train_batch_size / 256.
    if warmup_steps
        learning_rate = global_step / warmup_steps * scaled_lr
    else
        learning_rate = scaled_lr
    end

    #Cosine decay learning rate schedule
    total_steps = get_train_steps(num_examples)
    learning_rate = ifelse.(global_step < warmup_steps, learning_rate, tf.train_cosine_decay(
        scaled_lr, global_step - warmup_steps,total_steps - warmup_steps)
    )
    
    return learning_rate
end


"""
Linear head for linear evaluation.

Args:
x: hidden state tensor of shape (bsz, dim)
is_training: training 
num_classes: number of classes
use_bias: whether to use bias
use_bn: whether to use BN for output units

Returns:
logits of shape(bsz, num_classes)
"""
function linear_layer(x, is_training, num_classes, use_bias=true, use_bn=false)
    @assert ndims(shape(x)), shape(x)
    x = tf.layers.dense(inputs=x, units=num_classes, 
                        use_bias=use_bias and not use_bn,
                        kernel_initializer=tf.random_normal_initializer(stdev=.01))

    if use_bn
        x = resnet.batch_norm_relu(x, is_training, relu=false, center=use_bias)
    end
    x = tf.identity(x, "out")
    return x 
end


"""
Head for projecting hiddens to contrastive loss
"""
function projection_head(hiddens, is_training)
    if head_proj_mode == "none"
        hiddens = hiddens
    elseif head_proj_mode == "linear"
        hiddens = linear_layer(hiddens, is_training, head_proj_dim,
                    use_bias=false, use_bn=true)
    elseif head_proj_mode == "nonlinear"
        hiddens = linear_layer(hiddens, is_training, hiddens.shape[-1],
                    use_bias=true, use_bn=true)
        for j in range(1, num_nlh_layers+1)
            hiddens = relu(hiddens)
            hiddens = linear_layer(hiddens, is_training, head_proj_dim,
                    use_bias=false, use_bn=true)
        end
    else
        error("Unknown head projection mode")
    end
    return hiddens
end


function supervised_head(hiddens, num_classes, is_training)
    logits = linear_layer(hiddens, is_training, num_classes)
    #for var in tf.trainable_variables():
    #   tf.add_to_collection("trainable_variables_inblock_5", var)
    return logits 
end