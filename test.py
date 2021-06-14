

import torch 
import noise2self 
from PIL import Image 
import numpy as np 
from utils import psnr
import cv2 


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    

gray = True 
if gray == True:    
    c = 1
else:
    c = 3
    
model = noise2self.Dncnn(c).to(device)
model.load_state_dict(torch.load('model2.pt'))
model.eval()

img_orig = np.array(Image.open('./test.jpg'))
if gray == True:
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    img = np.array(Image.open('./noisy.jpg'))[:, :, 0, np.newaxis]
else:
    img = np.array(Image.open('./noisy.jpg'))


print(psnr(img / 255.0, img_orig/255.0))

img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
img_ten = torch.tensor(img).float().to(device) / 255.0 


out = model(img_ten)

out.clamp_(0, 1)

out_np = out[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
print(psnr(img_orig/255.0, out_np / 255.0))
if gray == True:
    out_np = np.repeat(out_np, 3, axis=2)
out_np = out_np.astype(np.uint8)
Image.fromarray(out_np).save('out.jpg')


