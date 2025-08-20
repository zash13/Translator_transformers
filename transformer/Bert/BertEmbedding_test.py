# this code complitly wroten by deepseek , whcih is nice case i dont know how to do it !!! : )
import tensorflow as tf
from BertEmbedding import preprocessing


vocab_size = 100
embedding_dim = 8
max_len = 50


embedding_layer = preprocessing(vocab_size, embedding_dim, max_len)


batch_size = 2
seq_length = 5


token_ids = tf.constant(
    [
        [12, 45, 3, 0, 88],
        [7, 31, 64, 2, 18],
    ],
    dtype=tf.int32,
)


segment_ids = tf.constant(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ],
    dtype=tf.int32,
)

print("=== INPUT ===")
print(f"Token IDs shape: {token_ids.shape}")
print(f"Token IDs:\n{token_ids.numpy()}\n")
print(f"Segment IDs shape: {segment_ids.shape}")
print(f"Segment IDs:\n{segment_ids.numpy()}\n")


inputs = (token_ids, segment_ids)
output = embedding_layer(inputs)

print("=== OUTPUT ===")
print(f"Output shape: {output.shape}")
print(f"Output embeddings for first sample:\n{output[0].numpy()}\n")
print(f"Output embeddings for second sample:\n{output[1].numpy()}\n")


print("=== DEBUGGING COMPONENTS ===")

token_embeddings = embedding_layer.tok_embedding(token_ids)
scaled_token_embeddings = token_embeddings * tf.sqrt(tf.cast(embedding_dim, tf.float32))
segment_embeddings = embedding_layer.seg_embedding(segment_ids)
positions = embedding_layer.pos_encoding[:, :seq_length, :]
position_embeddings = tf.tile(positions, [tf.shape(token_ids)[0], 1, 1])

print(f"Raw token embeddings shape: {token_embeddings.shape}")
print(f"Scaled token embeddings shape: {scaled_token_embeddings.shape}")
print(f"Segment embeddings shape: {segment_embeddings.shape}")
print(f"Position embeddings shape: {position_embeddings.shape}\n")

print("First token embedding (before scaling):", token_embeddings[0, 0].numpy())
print("First token embedding (after scaling):", scaled_token_embeddings[0, 0].numpy())
print("First segment embedding:", segment_embeddings[0, 0].numpy())
print("First position embedding:", position_embeddings[0, 0].numpy())


manual_sum = (
    scaled_token_embeddings[0, 0] + segment_embeddings[0, 0] + position_embeddings[0, 0]
)
print(f"\nManual sum for first position: {manual_sum.numpy()}")
print(f"Layer output for first position: {output[0, 0].numpy()}")
print(
    f"Are they equal? {tf.reduce_all(tf.abs(manual_sum - output[0, 0]) < 1e-6).numpy()}"
)
