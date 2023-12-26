import torch.nn.functional as F
import torch

def gentrator_loss(discriminator_generator_output, generator_output, target, lambda_=100):
    l1_loss = F.l1_loss(generator_output, target)
    gan_loss = F.binary_cross_entropy_with_logits(
                              discriminator_generator_output,
                              torch.ones_like(
                                   discriminator_generator_output)
                              )
    return l1_loss * lambda_ + gan_loss, gan_loss.item(), l1_loss.item()

def discriminator_loss(real_discriminator_output, fake_discriminator_output):
    real_loss = F.binary_cross_entropy_with_logits(real_discriminator_output,
                                                   torch.ones_like(real_discriminator_output)
                                                   )
    fake_loss = F.binary_cross_entropy_with_logits(fake_discriminator_output,
                                                   torch.zeros_like(fake_discriminator_output),
                                                   )
    return real_loss + fake_loss
