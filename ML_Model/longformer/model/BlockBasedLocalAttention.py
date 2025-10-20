import math
import tensorflow as tf
import pdb

def split_into_blocks(inputs, block_size):
    """
    Splits inputs into fixed-size blocks.
    Args:
        inputs: Tensor of shape [batch_size, seq_len, embed_dim]
        block_size: int, size of each block.
    Returns:
        Tensor of shape [batch_size, num_blocks, block_size, embed_dim]
    """
    print("####split_into_blocks")
    
    input_shape = inputs.shape 
    print("### input_shape", input_shape)
    
    batch_size = input_shape[0]
    print("### batch_size", batch_size)
    
    seq_len = input_shape[1]
    print("### seq_len", seq_len)
    
    embed_dim = input_shape[2]
    print("### embed_dim", embed_dim)

    num_blocks = math.ceil(seq_len // block_size)
    
    
    
    # Reshape padded inputs to split into blocks
    print("Reshape padded inputs to split into blocks")
    
    reshaped_inputs = tf.keras.layers.Reshape([num_blocks, block_size, embed_dim])(inputs)
    print("@@@@@")
    print(reshaped_inputs.shape)
    
    return reshaped_inputs

def get_block_context(blocks, left_context, right_context, block_size):
    """
    Extends blocks with specified left and right context.
    Args:
        blocks: Tensor of shape [batch_size, num_blocks, block_size, embed_dim]
        left_context: int, size of left context.
        right_context: int, size of right context.
    Returns:
        Tensor with context, shape [batch_size, num_blocks, block_size + left_context + right_context, embed_dim]
    """
    # batch_size, num_blocks, block_size, embed_dim = tf.shape(blocks)
    # batch_size = tf.shape(blocks)[0]
    # num_blocks = tf.shape(blocks)[1]  # Renamed for clarity
    # block_size = tf.shape(blocks)[2]
    # embed_dim = tf.shape(blocks)[3]
    print("get_block_context")
    num_blocks = tf.shape(blocks)[1]
    # tf.print("num_blocks", num_blocks)
    current_block_size = tf.shape(blocks)[2]  # This should match the provided 'block_size' if the function is used correctly
    # tf.print("current_block_size", current_block_size)
    embed_dim = tf.shape(blocks)[3]
    # tf.print("embed_dim", embed_dim)
    
    padded_blocks = tf.pad(blocks, [[0, 0], [left_context, right_context], [0, 0], [0, 0]])
    blocks_with_context = tf.TensorArray(dtype=blocks.dtype, size=num_blocks,dynamic_size=False)

    batch_size = 20

    for i in tf.range(num_blocks):
        # Calculate the start and end of the context window
        context_start = i
        context_end = i + left_context + right_context + 1  # +1 because slice end index is exclusive

        # Gather the context for block `i`
        context = padded_blocks[:, context_start:context_end, :, :]
        
        # Flatten the context dimensions into the block_size dimension
        context_reshaped = tf.reshape(context, [batch_size, -1, embed_dim])

        # Write the reshaped context to the tensor array
        blocks_with_context = blocks_with_context.write(i, context_reshaped)

    return blocks_with_context.stack()

def scaled_dot_product_attention(query, key, value):
    """
    Calculates the attention weights and returns the weighted sum of value vectors.
    Args:
        query, key, value: Tensors with shape [batch_size, ..., seq_len, depth].
    Returns:
        The weighted sum of value vectors.
    """
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

class BlockBasedLocalAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, block_size, left_context, right_context, **kwargs):
        super(BlockBasedLocalAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.left_context = left_context
        self.right_context = right_context
        self.query_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.key_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.value_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.final_dense = tf.keras.layers.Dense(embed_dim)

    def call(self, inputs):
        # Linear projections
        print("in call 1 ")
        
        queries = self.query_dense(inputs)
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)
        
        # Split into blocks
        # querey size is [batch_size * num_blocks * block_size * embed]
        query_blocks = split_into_blocks(queries, self.block_size)
        key_blocks = split_into_blocks(keys, self.block_size)
        value_blocks = split_into_blocks(values, self.block_size)

        # Extend blocks with context
        key_blocks_with_context = get_block_context(key_blocks, self.left_context, self.right_context, self.block_size)
        value_blocks_with_context = get_block_context(value_blocks, self.left_context, self.right_context, self.block_size)

        # Initialize tensor array to store attention output for each block
        # output_blocks = tf.TensorArray(dtype=tf.float32, size=tf.shape(query_blocks)[1])
        output_blocks = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # Loop through each block to compute attention
        for i in tf.range(tf.shape(query_blocks)[1]):
            block_output = scaled_dot_product_attention(
                query_blocks[:, i, :, :], 
                key_blocks_with_context[:, i, :, :], 
                value_blocks_with_context[:, i, :, :]
            )
            # output_blocks.append(block_output)
            output_blocks = output_blocks.write(i, block_output)

        # Concatenate the processed blocks back into a full sequence
        output_blocks_tensor = output_blocks.stack()
        # output = tf.concat(output_blocks, axis=1)
        # output = output_blocks.concat()

        # Reshape to match the input dimensions [batch_size, seq_len, embed_dim]
        output = tf.reshape(output_blocks_tensor, tf.shape(inputs))
        # output = tf.reshape(output, tf.shape(inputs))

        # Apply final dense layer
        print("Apply final dense layer")
        output = self.final_dense(output)

        return output