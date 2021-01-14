import tensorflow as tf
import tensorflow.keras.backend as K

def l1(x, y):
    x = K.reshape(x, (len(x), -1))
    y = K.reshape(y, (len(y), -1))
    return K.sum(K.abs(x - y), axis=1)  # keep batch dimension


# WGAN-GP version (dD/dx and dD/dz both regularized on interpolated inputs)
def gradient_penalty(discriminator, x, x_gen, z, z_gen, training):
    z_epsilon = K.random_uniform((len(x), 1), 0.0, 1.0)
    x_epsilon = K.reshape(z_epsilon, (len(x), 1, 1, 1))

    x_hat = x_epsilon * x + (1 - x_epsilon) * x_gen
    z_hat = z_epsilon * z + (1 - z_epsilon) * z_gen

    with tf.GradientTape() as t:
        t.watch([x_hat, z_hat])
        score_hat = discriminator([x_hat, z_hat], training=training)
    dx, dz = t.gradient(score_hat, [x_hat, z_hat])

    # flatten gradients, keep batch size
    dx = K.reshape(dx, (dx.shape[0], -1))
    dz = K.reshape(dz, (dz.shape[0], -1))
    grads = K.concatenate((dx, dz), axis=1)

    grads_norm = K.sqrt(K.sum(K.square(grads), axis=1))  # compute gradient norm
    # norm_penalty = K.square(K.relu(grads_norm - 1))  # norm should be <=1 (one-sided GP) 
    norm_penalty = K.square(grads_norm - 1)  # norm should be 1 (two-sided GP)
    return K.mean(norm_penalty)  # mean on batch


@tf.function
def train_step(images, generator, encoder, discriminator,
               generator_encoder_optimizer, discriminator_optimizer,
               d_train, ge_train, alpha=0.0001, gp_weight=2.5):

    actual_batch_size = len(images)
    latent_size = generator.input.shape[1]
    latent = K.random_normal([actual_batch_size, latent_size])

    with tf.GradientTape() as ge_tape, tf.GradientTape() as d_tape:
        # generation
        generated_images = generator(latent, training=ge_train)  # G(z)
        generated_latent = encoder(images, training=ge_train)  # E(x)

        reconstructed_images = generator(generated_latent, training=ge_train)  # G(E(x))
        reconstructed_latent = encoder(generated_images, training=ge_train)  # E(G(z))

        # discrimination
        real_score = discriminator([images, generated_latent], training=d_train)  # D(x, E(x))
        fake_score = discriminator([generated_images, latent], training=d_train)  # D(G(z), z)

        ### discriminator losses
        discriminator_loss = K.mean(fake_score - real_score)  # L_D

        # gradient penalty regularization
        gradient_penalty_loss = gradient_penalty(discriminator,
                                                 images, generated_images,
                                                 latent, generated_latent,
                                                 training=d_train)
        # total
        discriminator_total_loss = discriminator_loss + gp_weight * gradient_penalty_loss

        ### generator losses
        generator_encoder_loss = K.mean(real_score - fake_score)  # L_E,G

        # consistency losses
        images_reconstruction_loss = K.mean(l1(images, reconstructed_images))  # L_R
        latent_reconstruction_loss = K.mean(l1(latent, reconstructed_latent))  # L_R'
        consistency_loss = images_reconstruction_loss + latent_reconstruction_loss  # L_C

        # generator and encoder loss
        generator_encoder_total_loss = (1 - alpha) * generator_encoder_loss + alpha * consistency_loss  # L*_E,G

    # compute and apply gradients
    if d_train:
        discriminator_gradients = d_tape.gradient(discriminator_total_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    if ge_train:
        generator_encoder_variables = generator.trainable_variables + encoder.trainable_variables
        generator_encoder_gradients = ge_tape.gradient(generator_encoder_total_loss, generator_encoder_variables)
        generator_encoder_optimizer.apply_gradients(zip(generator_encoder_gradients, generator_encoder_variables))

    # compute mean scores for logging
    mean_real_score = K.mean(real_score)
    mean_fake_score = K.mean(fake_score)

    losses = {'generator_encoder_loss': generator_encoder_loss,
              'discriminator_loss': discriminator_loss,
              'images_reconstruction_loss': images_reconstruction_loss,
              'latent_reconstruction_loss': latent_reconstruction_loss,
              'gradient_penalty_loss': gradient_penalty_loss,
              'generator_encoder_total_loss': generator_encoder_total_loss,
              'discriminator_total_loss': discriminator_total_loss}

    scores = {'real_score': mean_real_score,
              'fake_score': mean_fake_score}

    return losses, scores
