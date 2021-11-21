# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal

def adaptive_resize(array, new_shape):
    # reshape the labels to the size of the image
    single_band = Image.fromarray(array)
    single_band_resized = single_band.resize(new_shape, Image.NEAREST)
    return np.asarray(single_band_resized)

  
def mask_landsat8_image_using_rasterized_shapefile(rasterized_shapefiles_path, district, this_landsat8_bands_list):
    this_shapefile_path = os.path.join(rasterized_shapefiles_path, "{}_shapefile.tif".format(district))
    ds = gdal.Open(this_shapefile_path)
    assert ds.RasterCount == 1
    shapefile_mask = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
    clipped_full_spectrum = list()
    for idx, this_band in enumerate(this_landsat8_bands_list):
        print("{}: Band-{} Size: {}".format(district, idx, this_band.shape))
        clipped_full_spectrum.append(np.multiply(this_band, shapefile_mask))
    x_prev, y_prev = clipped_full_spectrum[0].shape
    x_fixed, y_fixed = int(128 * np.ceil(x_prev / 128)), int(128 * np.ceil(y_prev / 128))
    diff_x, diff_y = x_fixed - x_prev, y_fixed - y_prev
    diff_x_before, diff_y_before = diff_x // 2, diff_y // 2
    clipped_full_spectrum_resized = [np.pad(x, [(diff_x_before, diff_x - diff_x_before), (diff_y_before, diff_y - diff_y_before)], mode='constant')
                                     for x in clipped_full_spectrum]
    print("{}: Generated Image Size: {}".format(district, clipped_full_spectrum_resized[0].shape, len(clipped_full_spectrum_resized)))
    return clipped_full_spectrum_resized
  
  
def get_images_from_large_file(bands, year, region, stride):
    # local machine
    data_directory_path = '/home/Projects/Forest/Data/all_billion_tree_regions/landsat-8/train_data'
    label_directory_path = '/home/Projects/Forest/Data/GroundTruth'
    destination = '/home/Projects/Forest/Data/generated_data'

    image_path = os.path.join(data_directory_path, 'landsat8_{}_region_{}.tif'.format(year,region))
    label_path = os.path.join(label_directory_path, '{}_{}.tif'.format(region, year))
    if not os.path.exists(destination):
        print('Log: Making parent directory: {}'.format(destination))
        os.mkdir(destination)
    print(image_path, label_path)
    # we will use this to divide those fnf images
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    # big_x_size, big_y_size = covermap.RasterXSize, covermap.RasterYSize
    label = channel.ReadAsArray()
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    x_size, y_size = image_ds.RasterXSize, image_ds.RasterYSize
    # we need the difference of the two raster sizes to do the resizing
    label = adaptive_resize(label, new_shape=(x_size, y_size))
    # print(label.shape, (y_size, x_size))
    all_raster_bands = [image_ds.GetRasterBand(x) for x in bands]
    count = 1
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            # read the label and drop this sample if it has all null pixels
            label_subset = label[i*stride:(i+1)*stride, j*stride:(j+1)*stride]
            if np.count_nonzero(label_subset) < 600:  # 0.01*256*256 ~ 650 pixels i.e at least 1% pixels should be valid
                print("(LOG): Dropping NULL Pixel Sample")
                continue
            # read the raster band by band for this subset
            example_subset = np.nan_to_num(all_raster_bands[0].ReadAsArray(j*stride, i*stride, stride, stride))
            for band in all_raster_bands[1:]:
                example_subset = np.dstack((example_subset, np.nan_to_num(band.ReadAsArray(j*stride, i*stride, stride, stride))))
            # save this example/label pair of numpy arrays as a pickle file with an index
            this_example_save_path = os.path.join(destination, '{}_{}_{}.pkl'.format(region, year, count))
            with open(this_example_save_path, 'wb') as this_pickle:
                pickle.dump((example_subset, label_subset), file=this_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                print('log: Saved {} '.format(this_example_save_path))
                print(i*stride, (i+1)*stride, j*stride, (j+1)*stride)
            count += 1
            
  def check_generated_dataset(path_to_dataset):
    for count in range(266):
        this_example_save_path = os.path.join(path_to_dataset, '{}.pkl'.format(count))
        with open(this_example_save_path, 'rb') as this_pickle:
            print('log: Reading {}'.format(this_example_save_path))
            (example_subset, label_subset) = pickle.load(this_pickle, encoding='latin1')
        show_image = np.asarray(255 * (example_subset[:, :, [4, 3, 2]] / 4096.0).clip(0, 1), dtype=np.uint8)
        plt.subplot(1,2,1)
        plt.imshow(show_image)
        plt.subplot(1,2,2)
        plt.imshow(label_subset)
        plt.show()
        
        
  def check_generated_fnf_datapickle(example_path):
    with open(example_path, 'rb') as this_pickle:
        (example_subset, label_subset) = pickle.load(this_pickle, encoding='latin1')
        example_subset = np.nan_to_num(example_subset)
        label_subset = fix(np.nan_to_num(label_subset))
    this = np.asarray(255*(example_subset[:,:,[3,2,1]]), dtype=np.uint8)
    that = label_subset
    plt.subplot(121)
    plt.imshow(this)
    plt.subplot(122)
    plt.imshow(that)
    plt.show()
    
    
    
def main():
    # generate pickle files to train from
    all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", 
                    "lower_dir", "malakand", "mansehra", "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    # number of images generated depends on value of stride
    for district in all_districts:
        get_images_from_large_file(bands=range(1, 12), year=2015, region=district, stride=256)
        
        
if __name__ == "__main__":
    main()