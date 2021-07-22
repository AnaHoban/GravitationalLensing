import os
import h5py
import shutil
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy import table
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#size
cutout_size = 64
nb_cutouts = 50000

#directories
scratch = os.path.expandvars("$SCRATCH") + '/'
tmp_dir = os.path.expandvars("$SLURM_TMPDIR") + '/'
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"

#all tiles
all_tiles = os.listdir(image_dir)

#tiles we can use
unused_tiles_file = './cutouts/list_unused_tiles.csv'
unused_tiles = list( (pd.read_csv(unused_tiles_file, dtype= str))['0'])

#getting cutouts
cutout_file = h5py.File(file_directory)

#create all cutout adresses
#all unused cutouts with both r and u channels
available_tiles_u = []
for tile in unused_tiles:
    r_tile = 'CFIS.' + tile + '.r.fits'
    u_tile = 'CFIS.' + tile + '.u.fits'
    if u_tile in all_tiles and r_tile in all_tiles: #taking cutouts with u and r bands
        available_tiles.append(tile)
        

#create a master_catalogue, this way we can keep track of the cutout name, it's tile and coordinates
#prepare master cat
example = table.Table.read(r_cat[1], format="ascii.sextractor")
keys = example.keys()
master_catalogue = pd.DataFrame(index = [0], columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'] + ['NB'])

print('master cat created \n')
#populate master cat

nb = 0
for tile_id in available_tiles: #single and both channels
    if count < nb_cutouts:
        rcat = table.Table.read(image_dir + 'CFIS.'+ tile_id + '.r' + ".cat", format="ascii.sextractor")
        
        u_img_dir = image_dir + 'CFIS.'+ tile_id + '.u' + ".fits"
        r_img_dir = image_dir + 'CFIS.'+ tile_id + '.r' + ".fits"
        
        # Copy tiles to $SLURM_TMPDIR
        shutil.copy2(u_img_dir, tmp_dir)
        shutil.copy2(r_img_dir, tmp_dir)

        count = 0
        for i in range(len(rcat)): #each cutout in tile
            if rcat["FLAGS"][i] != 0 or rcat["MAG_AUTO"][i] >= 99.0 or rcat["MAGERR_AUTO"][i] <= 0 or rcat["MAGERR_AUTO"][i] >= 1:
                continue
            
            #keep track
            new_cutout = pd.DataFrame(index = [i], data=np.array(rcat[i]), columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'] + ['NB'])
            new_cutout['BAND'] = 'r'
            new_cutout['TILE'] = tile
            new_cutout['CUTOUT'] = f"{count}"
            new_cutout['NB'] = f"c{nb}"

            master_catalogue = master_catalogue.append(new_cutout)
             
            count += 1
            nb += 1
    
#save
master_catalogue.to_csv(scratch + 'classify_catalogue.csv')     



#create cutouts function
def create_cutouts(img, x, y):
    ''' Creates the image and weight cutouts given a tile, the position of the center and the band '''
    
    img_cutout = Cutout2D(img.data, (x, y), cutout_size, mode="partial", fill_value=0).data
    
    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05*cutout_size**2 or np.count_nonzero(img_cutout) == 0: # Don't use this cutout
        return (np.zeros(cutout_size,cutout_size))
    
    img_cutout[np.isnan(img_cutout)] = 0
    
    img_lower = np.percentile(img_cutout, 0.001)
    img_upper = np.percentile(img_cutout, 99.999)
    
    img_cutout[img_cutout<img_lower] = img_lower
    img_cutout[img_cutout>img_upper] = img_upper
    
    if img_lower == img_upper:
        img_norm = np.zeros((cutout_size, cutout_size))
    else:
        img_norm = (img_cutout - np.min(img_cutout)) / (img_upper - img_lower)
        #inorms.append(1/(img_upper - img_lower))
        
    return img_norm
           
    
#batch them
def cutout_generator(hf, cut_list, batch_size = 256):
    cut_batch = np.zeros((batch_size, cutout_size, cutout_size, 4)) #3rd one is for number tracking
    
    while True:
        for tile in master_catalogue['TILE'].unique():
            u_name = image_dir + 'CFIS.'+ tile_id + '.u' + ".fits"
            r_name = image_dir + 'CFIS.'+ tile_id + '.r' + ".fits"
            
            #open fits file
            u_image = fits.open(tmp_dir + u_name, memmap=True)
            r_image = fits.open(tmp_dir + r_name, memmap=True)   
            
            for cutout in master_catalogue[master_catalogue['TILE'] == tile]['NB']:
                x = master_catalogue[master_catalogue['NB']==cutout]['X_IMAGE']
                y = master_catalogue[master_catalogue['NB']==cutout]['Y_IMAGE']
                
                cut_batch[i,...,0] = create_cutout(u_image, x, y)
                cut_batch[i,...,1] = create_cutout(r_image, x, y)           
                cut_batch[i,0,0,2] = cutout #tracking
                b += 1
                if b == batch_size:
                    b = 0
                    #yield (sources, sources)
                    yield cutouts
                    
            u_image.close()
            r_image.close()


#create the dataset
dataset = tf.data.Dataset.from_generator(cutout_generator, output_signature=(tf.TensorSpec(shape=(64,64,4), dtype=tf.float64)))
#save
tf.data.experimental.save(dataset, scratch + 'class_dataset')