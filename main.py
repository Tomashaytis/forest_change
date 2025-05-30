import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

CUR_MONTH_PATH = 'august'
PREV_MONTH_PATH = 'july'
MODEL_PATH = 'models\\model_random_forest_full.pkl'
MASKS_PATH = 'masks'
FEATURES = ['red', 'green', 'blue', 'delta_red', 'delta_green', 'delta_blue',
            'prev_ndvi', 'delta_ndvi', 'mean_delta_ndvi', 'prev_evi', 'delta_evi', 'mean_delta_evi']
TILE_SIZE = (512, 512)
THRESHOLD_PROB = 0.95


def convert_tile(rgb_cur: np.ndarray, ndvi_cur: np.ndarray, rgb_prev: np.ndarray, ndvi_prev: np.ndarray) \
        -> pd.DataFrame:
    """
    Converts tiles to DataFrame
    :param rgb_cur: current RGB tile
    :param ndvi_cur: current NDVI tile
    :param rgb_prev: current RGB tile
    :param ndvi_prev: current NDVI tile
    :return: tile DataFrame
    """
    delta_ndvi = ndvi_cur - ndvi_prev
    mean_delta_ndvi = delta_ndvi.mean()

    red_prev = rgb_prev[:, :, 0] / 255
    red_cur = rgb_cur[:, :, 0] / 255
    green_prev = rgb_prev[:, :, 1] / 255
    green_cur = rgb_cur[:, :, 1] / 255
    blue_prev = rgb_prev[:, :, 2] / 255
    blue_cur = rgb_cur[:, :, 2] / 255

    nir_prev = (red_prev * (ndvi_prev / 127 - 1) + red_prev) / (1 - (ndvi_prev / 127 - 1))
    nir_cur = (red_cur * (ndvi_cur / 127 - 1) + red_cur) / (1 - (ndvi_cur / 127 - 1))

    evi_prev = 2.5 * (nir_prev - red_prev) / (nir_prev + 6 * red_prev - 7.5 * blue_prev + 1)
    evi_cur = 2.5 * (nir_cur - red_cur) / (nir_cur + 6 * red_cur - 7.5 * blue_cur + 1)

    evi_prev = np.int32((evi_prev + 1) * 127)
    evi_cur = np.int32((evi_cur + 1) * 127)

    delta_evi = evi_cur - evi_prev
    mean_delta_evi = delta_evi.mean()

    length = rgb_cur.shape[0] * rgb_cur.shape[1]
    pixels = list(zip(np.ravel(red_cur), np.ravel(green_cur), np.ravel(blue_cur),
                      np.ravel(red_prev), np.ravel(green_prev), np.ravel(blue_prev),
                      np.ravel(ndvi_prev), np.ravel(delta_ndvi), [mean_delta_ndvi] * length,
                      np.ravel(evi_prev), np.ravel(delta_evi), [mean_delta_evi] * length))
    tile_df = pd.DataFrame(pixels, columns=['red', 'green', 'blue',
                                            'prev_red', 'prev_green', 'prev_blue',
                                            'prev_ndvi', 'delta_ndvi', 'mean_delta_ndvi',
                                            'prev_evi', 'delta_evi', 'mean_delta_evi'])
    tile_df['delta_red'] = tile_df['red'] - tile_df['prev_red']
    tile_df['delta_green'] = tile_df['green'] - tile_df['prev_green']
    tile_df['delta_blue'] = tile_df['blue'] - tile_df['prev_blue']

    return tile_df


def process_tile(tile_df: pd.DataFrame, features: list, model: RandomForestClassifier, tile_size = (512, 512), threshold_prob = 0.95):
    """
    Processes tile DataFrame ant returns binary mask with detected forest change
    :param tile_df: tile DataFrame
    :param features: features list
    :param model: forest change detection model
    :param tile_size: tile size
    :param threshold_prob: threshold probability
    :return: binary mask
    """
    y_pred = model.predict_proba(tile_df[features])[:,1]
    mask = np.reshape(y_pred, tile_size)
    return mask > threshold_prob


def main():
    tiles = os.listdir(PREV_MONTH_PATH)
    with open(MODEL_PATH, 'rb') as fp:
        model = pickle.load(fp)

    for tile in tqdm(tiles, desc='Process tiles'):
        rgb_cur = np.array(Image.open(os.path.join(CUR_MONTH_PATH, tile, 'RGB.tif'))).astype(np.int32)
        ndvi_cur = np.array(Image.open(os.path.join(CUR_MONTH_PATH, tile, 'NDVI.tif'))).astype(np.int32)
        rgb_prev = np.array(Image.open(os.path.join(PREV_MONTH_PATH, tile, 'RGB.tif'))).astype(np.int32)
        ndvi_prev = np.array(Image.open(os.path.join(PREV_MONTH_PATH, tile, 'NDVI.tif'))).astype(np.int32)

        tile_df = convert_tile(rgb_cur, ndvi_cur, rgb_prev, ndvi_prev)

        mask = process_tile(tile_df, FEATURES, model, TILE_SIZE, THRESHOLD_PROB)

        mask_uint8 = mask.astype(np.uint8) * 255
        img = Image.fromarray(mask_uint8)
        img.save(os.path.join(MASKS_PATH, f'{tile}.tif'))


if __name__ == '__main__':
    main()