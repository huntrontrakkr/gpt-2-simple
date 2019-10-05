import tensorflow as tf

from gpt_2_simple.src import model


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        pred=tf.equal(k, 0),
        true_fn=lambda: logits,
        false_fn=lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.compat.v1.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.compat.v1.where(probs_sums < p, logits_sort, tf.ones_like(
            logits_sort)*1000)  # [batchsize, vocab]
        min_logits = tf.reduce_min(input_tensor=logits_masked, axis=1, keepdims=True)  # [batchsize, 1]
        return tf.compat.v1.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(*, hparams, length, start_token=None,
                    batch_size=None, context=None, temperature=1,
                    top_k=0, top_p=0.0, truncate=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    if truncate is not None:
        truncate = tf.stack([truncate * batch_size], axis=0)

    compression_axes = [x for x in range(1, tf.rank(truncate))]

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens,
                                past=past, reuse=tf.compat.v1.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(
            hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output, inner_truncate):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(
                logits, num_samples=1, dtype=tf.int32)
            new_output = tf.concat([output, samples], axis=1)
            if truncate.shape[1] <= new_output.shape[1]:
                new_truncate = tf.reduce_all(tf.equal(new_output[:, -truncate.shape[1]:], truncate), axis=compression_axes)
            else:
                new_truncate = inner_truncate
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                new_output,
                tf.logical_or(inner_truncate, new_truncate)
            ]

        def cond(*args):
            return True

        def cond_truncate(*args):
            return not tf.reduce_all(args[3])

        _, _, tokens, _ = tf.while_loop(
            cond=cond if truncate is None else cond_truncate,
            body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                tf.zeros([batch_size], dtype=tf.dtypes.bool)
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(
                    hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size]),
            ],
            back_prop=False,
        )

        return tokens
