ENV["PYTHON"]= "/Users/serdar/miniconda/envs/tf/bin/python"

using Knet, IterTools
using PyCall

tf = pyimport("tensorflow")

BATCH_NORM_EPSILON = 1e-5

"""
Perform a batch normalization followed by a ReLU.

Args:
    inputs: 'Tensor' of shape [batch, channels, height, width]
    is_training: 'bool' for whether the model is training. 
    relu: 'bool' if False, skips ReLU operation.
    init_zero: 'bool' either "true" to initialize scale paramater with 0
            or "false" with 1. 
    center: 'bool' whether to add the learnable bias vector
    scale: 'bool' whether to add the learnable scaling factor 
    data_format: 'str' either "channels_first" for [batch, channels, height, width]
            or "channels_last" for [batch, height, width, channels]

Returns:
    A normalized 'Tensor' with the same data format.
"""
function batch_norm_relu(inputs, is_training; relu=true, init_zero=false,
        center=true, scale=true, data_format="channels_first")
if init_zero
    gamma_initializer = zeros()
else
    gamma_initializer = ones()
end

if data_format == "channels_first"
    axis = 1
else
    axis = 3
end

bn_foo = BatchNormalization(
        axis=axis,
        momentum=batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=scale,
        fused=false,
        gamma_initializer=gamma_initializer)
inputs = bn_foo(inputs, training=is_training)


if relu
    inputs=relu(inputs)
end
 
return input

"""
Pads the input along the spatial dimensions independently of input size

Args:
    inputs: 'Tensor' of size as 'data format'
    kernel_size: 'int' kernel size to be used for 'conv2d' or 'max_pool2d'
    operations. Should be positive integer.
    data_format: 'str' either "channels_first" for [batch, channels, height, width]
    or "channels_last" for [batch, height, width, channels]

Returns:
    A padded 'Tensor' of the same 'data_format' with size either intact
    (for 'kernel_size' = 1) or padded(for 'kernel_size' > 1)
"""
function fixed_padding(inputs, kernel_size, data_format="channels_first")
    pad_total = kernel_size - 1
    pad_beg = pad_total ÷ 2
    pad_end = pad_total - pad_beg

    ## TODO: tf.pad according to data_format

    return padded_inputs
end 

"""
Strided 2-D convolution with explicit padding.

The padding is consistent and is based only on 'kernel_size', not on the 
dimensions of 'inputs' (??as opposed to using 'tf.layers.conv2d' alone??)
Args:
    inputs: 'Tensor' of size as 'data format'
    kernel_size: 'int' kernel size to be used for 'conv2d' or 'max_pool2d'
    operations. Should be positive integer.
    data_format: 'str' either "channels_first" for [batch, channels, height, width]
    or "channels_last" for [batch, height, width, channels]

Returns:
 A 'Tensor' of shape [batch, filters, height_out, width_out]
"""
function conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                            data_format="channels_first")
    if strides > 1
        inputs = fixed_padding(inputs, kernel_size, data_format=data_format)
    end
    
    #inputs = conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size
    #                padding=('SAME' if strides==1 else 'VALID')),
    #                use_bias=false, kernel_initializer=variance_scaling_initializer(),
    #                data_format = data_format)

    return inputs




"""
Standard building block for residual networks with BN after convolutions

Args:
    inputs: 'Tensor' of size [batch, channels, height, width]
    filters: 'int' number of filters for the first two convolutions. The 
            third and final convolution will use 4 times as many filters.
    is_training= 'bool' for whether the model in training.
    strides= 'int' block stride. If greater than 1, the block will 
            downsample the input.
    use_projection= 'bool' for whether this block should use a use_projection
            shortcut (vs default identity shourtcut). This is usually 'True'
            for the first block of a block group, which may change the number
            of filters and the resolution.
    data_format= 'str' - either "channels_first" for [batch, channels, height,
             width] or  "channels_last" for [batch, height, width, channels]
Returns: 
    A 'Tensor' of shape [batch, filters, height_out, width_out]
"""
function residual_block(inputs, filters, is_training, strides, 
use_projection=false, data_format="channels_first")

shortcut = inputs 
if use_projection
    shortcut = conv2d_fixed_padding( inputs=inputs, filters=filters,
                 kernel_size=1, strides=strides,  data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=false, 
                data_format=data_format)
end


inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, 
                    kernel_size=3, strides=strides, data_format=data_format)
inputs =  batch_norm_relu(inputs, is_training, data_format=data_format)

inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, 
                        kernel_size=3, strides=1,data_format=data_format)
inputs =  batch_norm_relu(inputs, is_training, relu=false, init_zero=true,
                         data_format=data_format)

return relu(inputs + shortcut)

end
    
"""
Bottleneck block variant for residual networks with BN after convolutions.
"""
function bottleneck_block()

return relu(inputs + shortcut)

"""
Creates one group of blocks for the ResNet model.
"""
function block_group(inputs, filters, block_fn, blocks, strides, is_training,
                    name, data_format="channels_first")
    # Only the first block per block_group uses projection shortcut and strides.
    inputs = block_fn(inputs, filters, is_training, strides,
                    use_projection=true, data_format=data_format)
    
    for _ in range(1, blocks)
        inputs = block_fn(inputs, filters, is_training, 1, 
                        data_format=data_format)
    end

    return #tf.identity(inputs, name)
end



"""
Generator for ResNet v1 models.

Args:
block_fn: 'function' -  either 'residual block' or 'bottleneck_block'
layers: list of 4 'int's - number of blocks to include in each of the 4 block groups
width multiplier: 'int' -  filters multiplier for network
cifar_stem: 'bool' - if 'true', use 3x3 conv without strides or pooling as stem.
"""
function resnet_v1_generator(block_fn, layers, width_multiplier,
    cifar_stem=false, data_format="channel_last")
    """
    Creation of the model graph
    """
    function model(inputs,is_training)
        if cifar_stem
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=64*width_multiplier, 
                kernel_size=3, strides=1, data_format=data_format)
            # Need replacement of tf.identity
            #inputs = tf.identity(inputs, "initial_conv")
            inputs = batch_norm_relu(inputs, is_training,
                    data_format=data_format)
            #inputs = tf.identity(inputs, "initial_max_pool")
        else
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=64*width_multiplier, 
                kernel_size=7, strides=2, data_format=data_format)
            #inputs = tf. identity(inputs, "initial_conv")
            inputs = batch_norm_relu(inputs, is_training,
            data_format=data_format)
            #inputs = tf.identity(inputs, "initial_max_pool")

            inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3,
                            strides=2, padding="SAME", data_format=data_format)
            inputs = tf.identity(inputs, "initial_max_pool")
        
        end


        """
        Add new trainable variables for the immediated precedent block
        """
        function filter_trainable_varibles(trainable_variables, after_block)
        
        end

        """
        Put variables into graph collection
        """
        function add_to_collection(trainable_variables, prefix)
        
        end
    
        trainable_variables = {}
        filter_trainables_variables(trainable_variables, after_block=0)
        if train_mode == "finetune" && fine_tune_after_block == 0
            #inputs = tf.stop_gradient(inputs)
        end

        inputs = block_group(inputs=inputs, filters=64 * width_multiplier, 
                        block_fn=block_fn, blocks=layers[0], strides=1, 
                        is_training=is_training, name="block_group1",
                        data_format=data_format)
        
        filter_trainables_variables(trainable_variables, after_block=1)
        if train_mode == "finetune" && fine_tune_after_block == 1
            #inputs = tf.stop_gradient(inputs)
        end

        inputs = block_group(inputs=inputs, filters=128 * width_multiplier, 
                        block_fn=block_fn, blocks=layers[1], strides=2, 
                        is_training=is_training, name="block_group2",
                        data_format=data_format)

        filter_trainables_variables(trainable_variables, after_block=2)
        if train_mode == "finetune" && fine_tune_after_block == 2
            #inputs = tf.stop_gradient(inputs)
        end

        inputs = block_group(inputs=inputs, filters=256 * width_multiplier, 
                        block_fn=block_fn, blocks=layers[2], strides=2, 
                        is_training=is_training, name="block_group3",
                        data_format=data_format)

        filter_trainables_variables(trainable_variables, after_block=3)
        if train_mode == "finetune" && fine_tune_after_block == 3
            #inputs = tf.stop_gradient(inputs)
        end

        inputs = block_group(inputs=inputs, filters=512 * width_multiplier, 
                        block_fn=block_fn, blocks=layers[3], strides=2, 
                        is_training=is_training, name="block_group4",
                        data_format=data_format)


        filter_trainables_variables(trainable_variables, after_block=4)
        if train_mode == "finetune" && fine_tune_after_block == 4
            #inputs = tf.stop_gradient(inputs)
        end
        
        """
        The activation is 7x7 so this is a global average pool
        """
        pool_size = (inputs.shape[1], inputs.shape[2])
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=pool_size, strides=1, padding="VALID",
            data_format=data_format)
        inputs = tf.identity(inputs, "final_avg_pool")
        inputs = tf.squeeze(inputs, (1,2))

        add_to_collection(trainable_variables, "trainable_variables_inblock_")

        return inputs
    end

    return model
end


"""
Returns the ResNet model for a given size and number of output classes
"""
function resnet_v1(resnet_depth, width_multiplier, cifar_stem=false,
    data_format=data_format)

    model_params = {
        18: {"block": residual_block, "layers": [2,2,2,2]},
        34: {"block": residual_block, "layers": [3,4,6,3]},
        50: {"block": bottleneck_block, "layers": [3,4,6,3]},
        101: {"block": bottleneck_block, "layers": [3,4,23,3]},
        152: {"block": bottleneck_block, "layers": [3,8,36,3]},
        200: {"block": bottleneck_block, "layers": [3,24,36,3]},
    }

    if resnet_depth ∉ model_params
        error("Not a valid resnet_depth", resnet_depth)
    end

    params = model_params[resnet_depth]

    return resnet_v1_generator(params["block"], params["layers"],
            width_multiplier, cifar_stem=cifar_stem, data_format=data_format)
end
