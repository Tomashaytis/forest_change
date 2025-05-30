import os
import numpy as np
import pandas as pd
from PIL import Image
from random import randint

PATH_TO_TILES = 'D:\\Projects\\Others\\DataScience\\Forest Change\\Tiles_2024'


if __name__ == '__main__':
    tiles = os.listdir(PATH_TO_TILES)
    pixels = []

    pixels_per_month = 48

    for tile in tqdm(tiles):
        mask_prev = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'prev-extent.tif'))).astype(np.int32)
        mask_may = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'may-extent.tif'))).astype(np.int32)
        mask_june = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'june-extent.tif'))).astype(np.int32)
        mask_july = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'july-extent.tif'))).astype(np.int32)
        mask_august = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'august-extent.tif'))).astype(np.int32)
        mask_september = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'september-extent.tif'))).astype(
            np.int32)

        delta_mask_may = mask_may - mask_prev
        delta_mask_june = mask_june - mask_may
        delta_mask_july = mask_july - mask_june
        delta_mask_august = mask_august - mask_july
        delta_mask_september = mask_september - mask_august

        delta_mask_may[delta_mask_may < 0] = 0
        delta_mask_june[delta_mask_june < 0] = 0
        delta_mask_july[delta_mask_july < 0] = 0
        delta_mask_august[delta_mask_august < 0] = 0
        delta_mask_september[delta_mask_september < 0] = 0

        delta_mask_may = delta_mask_may.astype(np.bool)
        delta_mask_june = delta_mask_june.astype(np.bool)
        delta_mask_july = delta_mask_july.astype(np.bool)
        delta_mask_august = delta_mask_august.astype(np.bool)
        delta_mask_september = delta_mask_september.astype(np.bool)

        del mask_prev, mask_may, mask_june, mask_july, mask_august, mask_september

        ndvi_prev = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'NDVI_prev.tif'))).astype(np.int32)
        ndvi_may = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'NDVI_05.tif'))).astype(np.int32)
        ndvi_june = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'NDVI_06.tif'))).astype(np.int32)
        ndvi_july = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'NDVI_07.tif'))).astype(np.int32)
        ndvi_august = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'NDVI_08.tif'))).astype(
            np.int32)
        ndvi_september = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'NDVI_09.tif')).convert('L')).astype(
            np.int32)

        delta_ndvi_may = ndvi_may - ndvi_prev
        delta_ndvi_june = ndvi_june - ndvi_may
        delta_ndvi_july = ndvi_july - ndvi_june
        delta_ndvi_august = ndvi_august - ndvi_july
        delta_ndvi_september = ndvi_september - ndvi_august

        mean_delta_ndvi_may = delta_ndvi_may.mean()
        mean_delta_ndvi_june = delta_ndvi_june.mean()
        mean_delta_ndvi_july = delta_ndvi_july.mean()
        mean_delta_ndvi_august = delta_ndvi_august.mean()
        mean_delta_ndvi_september = delta_ndvi_september.mean()

        rgb_prev = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'RGB_prev.tif'))).astype(np.int32)
        rgb_may = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'RGB_05.tif'))).astype(np.int32)
        rgb_june = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'RGB_06.tif'))).astype(
            np.int32)
        rgb_july = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'RGB_07.tif'))).astype(
            np.int32)
        rgb_august = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'RGB_08.tif'))).astype(
            np.int32)
        rgb_september = np.array(Image.open(os.path.join(PATH_TO_TILES, tile, 'RGB_09.tif'))).astype(
            np.int32)

        red_prev = rgb_prev[:, :, 0] / 255
        red_may = rgb_may[:, :, 0] / 255
        red_june = rgb_june[:, :, 0] / 255
        red_july = rgb_july[:, :, 0] / 255
        red_august = rgb_august[:, :, 0] / 255
        red_september = rgb_september[:, :, 0] / 255

        green_prev = rgb_prev[:, :, 1] / 255
        green_may = rgb_may[:, :, 1] / 255
        green_june = rgb_june[:, :, 1] / 255
        green_july = rgb_july[:, :, 1] / 255
        green_august = rgb_august[:, :, 1] / 255
        green_september = rgb_september[:, :, 1] / 255

        blue_prev = rgb_prev[:, :, 2] / 255
        blue_may = rgb_may[:, :, 2] / 255
        blue_june = rgb_june[:, :, 2] / 255
        blue_july = rgb_july[:, :, 2] / 255
        blue_august = rgb_august[:, :, 2] / 255
        blue_september = rgb_september[:, :, 2] / 255

        nir_prev = (red_prev * (ndvi_prev / 127 - 1) + red_prev) / (1 - (ndvi_prev / 127 - 1))
        nir_may = (red_may * (ndvi_may / 127 - 1) + red_may) / (1 - (ndvi_may / 127 - 1))
        nir_june = (red_june * (ndvi_june / 127 - 1) + red_june) / (1 - (ndvi_june / 127 - 1))
        nir_july = (red_july * (ndvi_july / 127 - 1) + red_july) / (1 - (ndvi_july / 127 - 1))
        nir_august = (red_august * (ndvi_august / 127 - 1) + red_august) / (1 - (ndvi_august / 127 - 1))
        nir_september = ((red_september * (ndvi_september / 127 - 1) + red_september) /
                         (1 - (ndvi_september / 127 - 1)))

        evi_prev = 2.5 * (nir_prev - red_prev) / (nir_prev + 6 * red_prev - 7.5 * blue_prev + 1)
        evi_may = 2.5 * (nir_may - red_may) / (nir_may + 6 * red_may - 7.5 * blue_may + 1)
        evi_june = 2.5 * (nir_june - red_june) / (nir_june + 6 * red_june - 7.5 * blue_june + 1)
        evi_july = 2.5 * (nir_july - red_july) / (nir_july + 6 * red_july - 7.5 * blue_july + 1)
        evi_august = (2.5 * (nir_august - red_august) /
                      (nir_august + 6 * red_august - 7.5 * blue_august + 1))
        evi_september = (2.5 * (nir_september - red_september) /
                         (nir_september + 6 * red_september - 7.5 * blue_september + 1))

        evi_prev = np.int32((evi_prev + 1) * 127)
        evi_may = np.int32((evi_may + 1) * 127)
        evi_june = np.int32((evi_june + 1) * 127)
        evi_july = np.int32((evi_july + 1) * 127)
        evi_august = np.int32((evi_august + 1) * 127)
        evi_september = np.int32((evi_september + 1) * 127)

        delta_evi_may = evi_may - evi_prev
        delta_evi_june = evi_june - evi_may
        delta_evi_july = evi_july - evi_june
        delta_evi_august = evi_august - evi_july
        delta_evi_september = evi_september - evi_august

        mean_delta_evi_may = delta_evi_may.mean()
        mean_delta_evi_june = delta_evi_june.mean()
        mean_delta_evi_july = delta_evi_july.mean()
        mean_delta_evi_august = delta_evi_august.mean()
        mean_delta_evi_september = delta_evi_september.mean()

        rand_mask = np.zeros((512, 512), np.bool)
        rand_pixels = []
        for i in range(pixels_per_month):
            while True:
                rand_pixel = randint(0, 511), randint(0, 511)
                if rand_pixel not in rand_pixels and not delta_mask_may[rand_pixel[0], rand_pixel[1]]:
                    rand_mask[rand_pixel[0], rand_pixel[1]] = True
                    rand_pixels.append(rand_pixel)
                    break

        coordinates = np.argwhere(rand_mask)
        selected_delta_ndvi = delta_ndvi_may[rand_mask]
        selected_ndvi = ndvi_prev[rand_mask]
        selected_delta_evi = delta_evi_may[rand_mask]
        selected_evi = evi_prev[rand_mask]
        length = len(selected_ndvi)
        pixels.extend(list(zip([tile] * length, coordinates[:, 0], coordinates[:, 1],
                               ['may'] * length, [False] * length,
                               red_may[rand_mask], green_may[rand_mask], blue_may[rand_mask],
                               red_prev[rand_mask], green_prev[rand_mask], blue_prev[rand_mask],
                               selected_ndvi, selected_delta_ndvi, [mean_delta_ndvi_may] * length,
                               selected_evi, selected_delta_evi, [mean_delta_evi_may] * length)))

        rand_mask = np.zeros((512, 512), np.bool)
        rand_pixels = []
        for i in range(pixels_per_month):
            while True:
                rand_pixel = randint(0, 511), randint(0, 511)
                if rand_pixel not in rand_pixels and not delta_mask_june[rand_pixel[0], rand_pixel[1]]:
                    rand_mask[rand_pixel[0], rand_pixel[1]] = True
                    rand_pixels.append(rand_pixel)
                    break

        coordinates = np.argwhere(rand_mask)
        selected_delta_ndvi = delta_ndvi_june[rand_mask]
        selected_ndvi = ndvi_may[rand_mask]
        selected_delta_evi = delta_evi_june[rand_mask]
        selected_evi = evi_may[rand_mask]
        length = len(selected_ndvi)
        pixels.extend(list(zip([tile] * length, coordinates[:,0], coordinates[:,1],
                               ['june'] * length, [False] * length,
                               red_june[rand_mask], green_june[rand_mask], blue_june[rand_mask],
                               red_may[rand_mask], green_may[rand_mask], blue_may[rand_mask],
                               selected_ndvi, selected_delta_ndvi, [mean_delta_ndvi_june] * length,
                               selected_evi, selected_delta_evi, [mean_delta_evi_june] * length)))

        rand_mask = np.zeros((512, 512), np.bool)
        rand_pixels = []
        for i in range(pixels_per_month):
            while True:
                rand_pixel = randint(0, 511), randint(0, 511)
                if rand_pixel not in rand_pixels and not delta_mask_july[rand_pixel[0], rand_pixel[1]]:
                    rand_mask[rand_pixel[0], rand_pixel[1]] = True
                    rand_pixels.append(rand_pixel)
                    break

        coordinates = np.argwhere(rand_mask)
        selected_delta_ndvi = delta_ndvi_july[rand_mask]
        selected_ndvi = ndvi_june[rand_mask]
        selected_delta_evi = delta_evi_july[rand_mask]
        selected_evi = evi_june[rand_mask]
        length = len(selected_ndvi)
        pixels.extend(list(zip([tile] * length, coordinates[:,0], coordinates[:,1],
                               ['july'] * length, [False] * length,
                               red_july[rand_mask], green_july[rand_mask], blue_july[rand_mask],
                               red_june[rand_mask], green_june[rand_mask], blue_june[rand_mask],
                               selected_ndvi, selected_delta_ndvi, [mean_delta_ndvi_july] * length,
                               selected_evi, selected_delta_evi, [mean_delta_evi_july] * length)))

        rand_mask = np.zeros((512, 512), np.bool)
        rand_pixels = []
        for i in range(pixels_per_month):
            while True:
                rand_pixel = randint(0, 511), randint(0, 511)
                if rand_pixel not in rand_pixels and not delta_mask_august[rand_pixel[0], rand_pixel[1]]:
                    rand_mask[rand_pixel[0], rand_pixel[1]] = True
                    rand_pixels.append(rand_pixel)
                    break

        coordinates = np.argwhere(rand_mask)
        selected_delta_ndvi = delta_ndvi_august[rand_mask]
        selected_ndvi = ndvi_july[rand_mask]
        selected_delta_evi = delta_evi_august[rand_mask]
        selected_evi = evi_july[rand_mask]
        length = len(selected_ndvi)
        pixels.extend(list(zip([tile] * length, coordinates[:,0], coordinates[:,1],
                               ['august'] * length, [False] * length,
                               red_august[rand_mask], green_august[rand_mask], blue_august[rand_mask],
                               red_july[rand_mask], green_july[rand_mask], blue_july[rand_mask],
                               selected_ndvi, selected_delta_ndvi, [mean_delta_ndvi_august] * length,
                               selected_evi, selected_delta_evi, [mean_delta_evi_august] * length)))

        rand_mask = np.zeros((512, 512), np.bool)
        rand_pixels = []
        for i in range(pixels_per_month):
            while True:
                rand_pixel = randint(0, 511), randint(0, 511)
                if rand_pixel not in rand_pixels and not delta_mask_september[rand_pixel[0], rand_pixel[1]]:
                    rand_mask[rand_pixel[0], rand_pixel[1]] = True
                    rand_pixels.append(rand_pixel)
                    break

        coordinates = np.argwhere(rand_mask)
        selected_delta_ndvi = delta_ndvi_september[rand_mask]
        selected_ndvi = ndvi_august[rand_mask]
        selected_delta_evi = delta_evi_september[rand_mask]
        selected_evi = evi_august[rand_mask]
        length = len(selected_ndvi)
        pixels.extend(list(zip([tile] * length, coordinates[:,0], coordinates[:,1],
                               ['september'] * length, [False] * length,
                               red_september[rand_mask], green_september[rand_mask], blue_september[rand_mask],
                               red_august[rand_mask], green_august[rand_mask], blue_august[rand_mask],
                               selected_ndvi, selected_delta_ndvi, [mean_delta_ndvi_september] * length,
                               selected_evi, selected_delta_evi, [mean_delta_evi_september] * length)))

    pixels = pd.DataFrame(pixels, columns=['tile', 'x', 'y', 'month', 'forest_change',
                                           'red', 'green', 'blue', 'prev_red', 'prev_green', 'prev_blue',
                                           'prev_ndvi', 'delta_ndvi', 'mean_delta_ndvi',
                                           'prev_evi', 'delta_evi', 'mean_delta_evi'])
    pixels.to_csv('data/false_pixels.csv')