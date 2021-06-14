
import torch 
import noise2self 
from PIL import Image 
import numpy as np 
from utils import psnr
import cv2 
import idx2numpy


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
imagefile = 'train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

    
model = noise2self.Dncnn(1).to(device)
model.load_state_dict(torch.load('model-mnist.pt'))
model.eval()

img_orig = imagearray[0][:, :, np.newaxis]


   
img = np.array(Image.open('./mnist-noisy.jpg'))[:, :, 0, np.newaxis]
print(img.shape)

print(psnr(img / 255.0, img_orig/255.0))

img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]

img_ten = torch.tensor(img).float().to(device) / 255.0 


out = model(img_ten)

out.clamp_(0, 1)

out_np = out[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
print(psnr(img_orig/255.0, out_np / 255.0))

out_np = np.repeat(out_np, 3, axis=2)
out_np = out_np.astype(np.uint8)
Image.fromarray(out_np).save('mnist-out.jpg')


