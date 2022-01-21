"""
Contrastive loss functions.
"""

LARGE_NUM = 1e9

"""
Compute loss for model and add it to loss collection
"""
function add_supervised_loss(labels, logits, weights, kwargs...)
    return tf.losses_softmax_cross_entropy(labels, logits, weights, kwargs...)
end

"""
Compute loss for model.

Args:
hidden: hidden vector 'Tensor' of shape (bsz, dim)
hidden_norm: whether or not to use normalization on the hidden vector.
temperature: a 'floating' number for temperature scaling.
weights: a weighting number or vector.

Returns:
A loss scalar.
The logits for contrastive prediction task.
The labels for contrastive prediction task.
"""
function add_contrastive_loss(hidden, hidden_norm=true, temperature=1.0, weights=1.0)
# Get (normalized) hidden1 and hidden2.
    if hidden_norm
        hidden = tf.math.l2_normalize(hidden, -1)
    end
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size*2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=true) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=true) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=true) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=true) / temperature

    loss_a = tf.losses.losses_softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa],1), weights=weights
    )

    loss_b = tf.losses.losses_softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb],1), weights=weights
    )

    loss = loss_a + loss_b

    return loss, logits_ab, labels
end