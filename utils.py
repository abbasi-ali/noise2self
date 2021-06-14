

import numpy as np 
from PIL import Image 

def psnr(x, y):
    return 10 * np.log10(1 / (((x - y)**2).mean()))

def random_grid(img, grid_num):
    h, w, c = img.shape 
    ys = np.linspace(0, h, grid_num + 1)[:-1]
    xs = np.linspace(0, w, grid_num + 1)[:-1]
    delta_y = int(h / grid_num)
    delta_x = int(w / grid_num)
    
    coords = [(int(y), int(x)) for y in ys for x in xs]
    grid_id = np.random.randint(0, grid_num ** 2) 
    y, x = coords[grid_id]
    
    tmp = img.copy()
    tmp[y:y+delta_y, x:x+delta_x, :] = np.random.randint(0, 255, (delta_y, delta_x, c))
    
    msk = np.zeros((h, w, c))
    msk[y:y+delta_y, x:x+delta_x, :] = 1
    
    return tmp, msk 


def random_grid2(img, grid_num):
    h, w, c = img.shape 
    ys = np.linspace(0, h, grid_num + 1)[:-1]
    xs = np.linspace(0, w, grid_num + 1)[:-1]
    delta_y = int(h / grid_num)
    delta_x = int(w / grid_num)
    
    coords = [(int(y), int(x)) for y in ys for x in xs]
    tmp = img.copy()
    msk = np.zeros((h, w, c))    
    # print(tmp.shape)
    res = np.random.randint(0, 2)
    for cnt in range(len(coords)):
        y_, x_ = coords[cnt]
        if cnt % 2 == res:
            # tmp[y_:y_+delta_y, x_:x_+delta_x, :] = np.random.randint(0, 255, (delta_y, delta_x, c))
            # msk[y_:y_+delta_y, x_:x_+delta_x, :] = 1          
            
            tmp[int(y_+0.9*delta_y):y_+delta_y, int(x_+0.9*delta_x):x_+delta_x, :] = np.random.randint(0, 255, (int(0.1*delta_y)+1, int(0.1*delta_x)+1, c))
            msk[int(y_+0.9*delta_y):y_+delta_y, int(x_+0.9*delta_x):x_+delta_x, :] = 1          
    
    return tmp, msk 

def random_grid_mnist(img, grid_num):
    h, w, c = img.shape 
    ys = np.linspace(0, h, grid_num + 1)[:-1]
    xs = np.linspace(0, w, grid_num + 1)[:-1]
    delta_y = int(h / grid_num)
    delta_x = int(w / grid_num)
    
    coords = [(int(y), int(x)) for y in ys for x in xs]
    
    tmp = img.copy()
    msk = np.zeros((h, w, c))    
    # print(tmp.shape)
    res = np.random.randint(0, 2)
    for cnt in range(len(coords)):
        y_, x_ = coords[cnt]
        if cnt % 2 == res:
            # tmp[y_:y_+delta_y, x_:x_+delta_x, :] = np.random.randint(0, 255, (delta_y, delta_x, c))
            # msk[y_:y_+delta_y, x_:x_+delta_x, :] = 1          
            
            tmp[y_:y_+delta_y, x_:x_+delta_x, :] = np.random.randint(0, 255, (delta_y, delta_x, c))
            msk[y_:y_+delta_y, x_:x_+delta_x, :] = 1          
    
    return tmp, msk 

def random_grid3(img, grid_num, grid_id):
    h, w, c = img.shape 
    ys = np.linspace(0, h, grid_num + 1)[:-1]
    xs = np.linspace(0, w, grid_num + 1)[:-1]
    delta_y = int(h / grid_num)
    delta_x = int(w / grid_num)
    
    coords = [(int(y), int(x)) for y in ys for x in xs]
    # grid_id = np.random.randint(0, grid_num ** 2) 
    y, x = coords[grid_id]
    
    tmp = img.copy()
    tmp[y:y+delta_y, x:x+delta_x, :] = np.random.randint(0, 255, (delta_y, delta_x, c))
    
    msk = np.zeros((h, w, c))
    msk[y:y+delta_y, x:x+delta_x, :] = 1
    
    return tmp, msk 


# img = np.array(Image.open('./test.jpg')) / 255.0 
# img, _  = random_grid2(img, 11)

# img = img * 255.0 
# img = img.astype(np.uint8)
# Image.fromarray(img).save('dfdfdfdfdfdfdfdf.jpg')


def add_noise_online(img):
    h, w, c = img.shape 
    img_tmp = img.copy() / 255.0
    # std = np.random.uniform(0.05, 0.3)
    noise = np.random.normal(0, 0.5, (h, w, c))
    noisy_img = (noise + img_tmp ).clip(0, 1)
    
    # print(utils.psnr(noisy_img, img))
    
    noisy_img = (noisy_img * 255.0).astype(np.uint8)
    return noisy_img
    
    

def add_noise_combine(img):
    h, w, c = img.shape 
    img_tmp = img.copy() 
    
    noise_g = np.random.normal(0, 80, (h, w, c))
    noise_p = np.random.poisson(30, (h, w, c))
    
    size = h * w * c 
    all_coords = [i for i in range(size)]
    samples_0 = np.random.choice(all_coords, int(size * 0.1))
    samples_1 = np.random.choice(all_coords, int(size * 0.1))
    
    noisy_img = (noise_g + img_tmp + noise_p ).clip(0, 255).reshape(-1)
    noisy_img[samples_0] = 0
    noisy_img[samples_1] = 255
    
    noisy_img = noisy_img.reshape((h, w, c))
    
    
    # print(psnr(noisy_img / 255.0, img / 255.0))
    
    noisy_img = noisy_img.astype(np.uint8)
    return noisy_img
    
# img = np.array(Image.open('./test.jpg')) 
# img = add_noise_combine(img)    
# Image.fromarray(img).save('test_combine.jpg')































