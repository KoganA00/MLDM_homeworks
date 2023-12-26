import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.transforms.functional as F_


#from losses import *

def train_g(generator,
          n_epochs,
          generator_opt,
          train_loader,
          val_loader=None,
          num_model=0,
          best_loss=None,
          logging=False,
          path_last_epoch='only_gen_',
          path_best_model='only_best_gen_'
          ):
    best_loss = best_loss
    generator.train()
    generator = generator.to(DEVICE)


    for epoch in range(n_epochs):
        loss = train_epoch_g(generator,
                                 generator_opt,
                                 train_loader,
                                 logging)
        print('\nepoch ' + str(epoch) + '/' + str(n_epochs),  loss )
        torch.save(generator.state_dict(),  path_last_epoch + str(num_model) +'.pt')
        if logging:
            wandb.log({
                'train_loss' : loss
            })
        if val_loader is not None:
            loss = validation_g(generator,
                                    val_loader,
                                    logging)
            print('\nvalidation ' + str(epoch) + '/' + str(n_epochs),  loss )
            if logging:
                wandb.log({
                   'val_loss' : loss
            })
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(generator.state_dict(), path_best_model + str(num_model) +'.pt')



def train_epoch_g(generator,
                opt_generator,
                train_loader,
                logging=False):


    loss_ = 0
    step_ = 0
    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        print(step_, end='\t')
        step_ += 1


        opt_generator.zero_grad()

        fake = generator(x)
        loss = F.l1_loss(fake, y)

        loss_ += loss.item()
        if logging:
            wandb.log({
                'train_step_loss' : loss.item()
            })
        loss.backward()
        opt_generator.step()

    if logging:
        wandb.log({
          'train_real' :  wandb.Image(F_.to_pil_image(x[0])),
          'train_truth' :  wandb.Image(F_.to_pil_image(y[0])),
          'train_fake' :  wandb.Image(F_.to_pil_image(fake[0]))
      })

    return loss_ / len(train_loader)


def validation_g(generator,
                val_loader,
                 logging=False):


    loss_ = 0
    step_ = 0
    for x, y in val_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        print(step_, end='\t')
        step_ += 1
        fake = generator(x)
        loss = F.l1_loss(fake, y)
        loss_ += loss.item()

        if logging:
            wandb.log({
                'val_step_loss' : loss.item()
            })

    if logging:
        wandb.log({
          'val_real' :  wandb.Image(F_.to_pil_image(x[0])),
          'val_truth' :  wandb.Image(F_.to_pil_image(y[0])),
          'val_fake' :  wandb.Image(F_.to_pil_image(fake[0]))
      })
    return loss_ / len(val_loader)



def train(generator,
          discriminator,
          n_epochs,
          generator_opt,
          discriminator_opt,
          train_loader,
          val_loader=None,
          num_model=0,
          best_loss=None,
          logging=False,
          path_last_epoch='pix2pix'
          ):
    best_loss = best_loss

    generator.train()
    generator = generator.to(DEVICE)

    discriminator.train()
    discriminator = discriminator.to(DEVICE)


    for epoch in range(n_epochs):
        g_loss, d_loss, gan_loss, l1_loss = train_epoch(generator,
                                       discriminator,
                                       generator_opt,
                                       discriminator_opt,
                                       train_loader,
                                       logging)
        print('\nepoch ' + str(epoch) + '/' + str(n_epochs), 'g loss', g_loss, 'd_loss' , d_loss)
        torch.save(generator.state_dict(), path_last_epoch+'/gen_' + str(num_model) +'.pt')
        torch.save(discriminator.state_dict(), path_last_epoch+'/dis_' + str(num_model) +'.pt')

        if logging:
            wandb.log({
                'train_g_loss' : g_loss,
                'train_d_loss' : d_loss,
                'train_gan_loss' : gan_loss,
                'train_l1_loss' : l1_loss

            })
        if val_loader is not None:
            g_loss, d_loss, gan_loss, l1_loss = validation(generator,
                                    discriminator,
                                    val_loader,
                                    logging)
            print('\nval ' + str(epoch) + '/' + str(n_epochs), 'g loss', g_loss, 'd_loss', d_loss )
            if logging:
                wandb.log({
                   'val_g_loss' : g_loss,
                   'val_d_loss' : d_loss,
                   'val_gan_loss' : gan_loss,
                   'val_l1_loss' : l1_loss
            })



def train_epoch(generator,
                discriminator,
                opt_generator,
                opt_discriminator,
                train_loader,
                logging=False):


    g_loss_ = 0
    d_loss_ = 0
    gan_loss_ = 0
    l1_loss_ = 0
    step_ = 0
    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        print(step_, end='\t')
        step_ += 1


        #Disriminator step#
        #Generator does NOT change
        #########
        for param in discriminator.parameters():
            param.require_grad = True
        for param in generator.parameters():
            param.require_grad = False
        opt_discriminator.zero_grad()

        discr_output_real = discriminator(y, x)
        fake = generator(x)
        #detch because we do not need gradient from generator

        discr_output_fake = discriminator(fake.detach(), x)

        loss_discriminator = discriminator_loss(discr_output_real,
                                                discr_output_fake) * 0.5
        real_loss_disc = loss_discriminator.item() * 2
        d_loss_ += real_loss_disc
        loss_discriminator.backward()
        opt_discriminator.step()
        #########

        if logging:
            wandb.log({
                'train_step_d_loss' : real_loss_disc
            })

        #Generator step#
        #Disriminator does NOT change
        #########
        for param in generator.parameters():
            param.require_grad = True
        for param in discriminator.parameters():
            param.require_grad = False
        opt_generator.zero_grad()


        discr_output_fake = discriminator(fake, x)

        loss_generator, loss_gan, loss_l1 = gentrator_loss(discr_output_fake, fake, y)

        g_loss_ += loss_generator.item()
        gan_loss_ += loss_gan
        l1_loss_ += loss_l1


        loss_generator.backward()
        opt_generator.step()
        ##########

        if logging:
            wandb.log({
                'train_step_g_loss' : loss_generator.item(),
                'train_step_gan_loss':loss_gan,
                'train_step_l1_loss':loss_l1
            })





    if logging:
        wandb.log({
          'train_real' :  wandb.Image(F_.to_pil_image(x[0] * 0.5 + 0.5)),
          'train_truth' :  wandb.Image(F_.to_pil_image(y[0] * 0.5 + 0.5)),
          'train_fake' :  wandb.Image(F_.to_pil_image(fake[0] * 0.5 + 0.5))
      })

    return g_loss_ / len(train_loader), d_loss_ / len(train_loader), gan_loss_ / len(train_loader), l1_loss_ / len(train_loader)


def validation(generator,
                 discriminator,
                 val_loader,
                 logging=False):


    g_loss_, d_loss_ = 0, 0
    step_ = 0
    gan_loss_ = 0
    l1_loss_ = 0

    generator.eval()
    generator = generator.to(DEVICE)

    discriminator.eval()
    discriminator = discriminator.to(DEVICE)

    with torch.no_grad():
        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            print(step_, end='\t')
            step_ += 1

            discr_output_real = discriminator(y, x)
            fake = generator(x)
            discr_output_fake = discriminator(fake, x)

            loss_discriminator = discriminator_loss(discr_output_real,
                                                discr_output_fake)
            d_loss_ += loss_discriminator.item()

            loss_generator, loss_gan, loss_l1 = gentrator_loss(discr_output_fake, fake, y)

            g_loss_ += loss_generator.item()
            gan_loss_ += loss_gan
            l1_loss_ += loss_l1

            if logging:
                wandb.log({
                    'val_step_d_loss' : loss_discriminator.item(),
                    'val_step_g_loss' : loss_generator.item(),
                    'val_step_gan_loss':loss_gan,
                    'val_step_l1_loss':loss_l1
                })

        if logging:
            wandb.log({
              'val_real' :  wandb.Image(F_.to_pil_image(x[0] * 0.5 + 0.5)),
              'val_truth' :  wandb.Image(F_.to_pil_image(y[0] * 0.5 + 0.5)),
              'val_fake' :  wandb.Image(F_.to_pil_image(fake[0] * 0.5 + 0.5))
          })
    return g_loss_ / len(val_loader), d_loss_ / len(val_loader), gan_loss_ / len(val_loader), l1_loss_ / len(val_loader)
