import os
import PIL
import tqdm 
import torch
import torchvision.transforms.functional as F_

def save_validation_results_to_folders(g, val_loader, fake_folder, real_folder):
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  g.eval()
  g =  g.to(DEVICE)

  if not os.path.exists(fake_folder):
    os.makedirs(fake_folder)

  if not os.path.exists(real_folder):
      os.makedirs(real_folder)

      
  ii=0
  for x, real in val_loader:
      fake = g(x.to(DEVICE))
      
      fake_img = F_.to_pil_image(fake[0] * 0.5 + 0.5)
      real_img = F_.to_pil_image(real[0] * 0.5 + 0.5)
      fake_img.save(fake_folder + str(ii) + '.jpg')
      real_img.save(real_folder + str(ii) + '.jpg')
      ii += 1
  

def depict_input_real_fake(generator, x, y, norm=False):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    generator = generator.to(DEVICE) 
    generator.eval()
    with torch.no_grad():
        fake = generator(x).detach().cpu()
  
 
    display_list = [x[0], y[0], fake[0]]

    new_img = PIL.Image.new('RGB',(256*3,256))

    for i in range(3):
        # Getting the pixel values in the [0, 1] range to plot.
        if not norm:
            new_img.paste(F_.to_pil_image(display_list[i]), (256*i, 0))
        else:
            new_img.paste(F_.to_pil_image(display_list[i] * 0.5 + 0.5), (256*i, 0))

        
    return new_img