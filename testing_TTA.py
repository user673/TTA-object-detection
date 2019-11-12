# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os

from TTA_final import TTA


model_path = os.path.join( 'IDS_4_it_c.h5')
# load retinanet model
model = models.load_model(model_path)

# load label to names mapping for visualization purposes
labels_to_names = { 0: 'Borjomi_glass',
                    1: 'morshin_ pine_water_lemon',
                    2: 'morshin_ pine_water_mint',
                    3: 'morshin_ pine_water_ bilberry',
                    4: 'Borjomi_plastic',
                    5: 'aqua_nyanya_5',
                    6: 'aqua_nyanya_033',
                    7: 'aqua_nyanya_sport',
                    8: 'morshin_peppa_pig',
                    9: 'morshin_princess',
                    10: 'morshin_spiderman',
                    11: 'morshin_b/g_033',
                    12: 'morshin_sport',
                    13: 'morshin_b/g',
                    14: 'truskavetska_g_1_5',
                    15: 'morshin_b/g_1_5',
                    16: 'morshin_low/g_1_5',
                    17: 'morshin_g',
                    18: 'mirgorod_05',
                    19: 'mirgorod_lagidna_05',
                    20: 'morshin_Junior',
                    21: 'truskavetska_g_05',
                    22: 'mirgorod_1_5',
                    23: 'mirgorod_lagidna_1_5',
                    24: 'morshin_premium_b/g',
                    25: 'Borjomi_plastic_2',
                    26: 'borjomi_g/b',
                    27: 'morshin_6',
                    28: 'morshinka_6',
                    29: 'aqua_nyanya_1_5',
                    30: 'morshin_low/g', 31: 'aqualife_6',
                    32: 'morshin_b/g_1_5_3+1promo_pack',
                    33: 'morshin_low/g_1_5_3+1promo_pack',
                    34: 'morshinka_1_5',
                    35: 'truskavetska_low/g_1_5',
                    36: 'truskavetska_b/g_1_5',
                    37: 'morshin_3',
                    38: 'morshin_sportik_033',
                    39: 'morshinka_sport',
                    40: 'mirgorod_1_5_6pack',
                    41: 'mirgorod_lagidna_1_5_6pack',
                    42: 'morshin_premium_low/g',
                    43: 'morshin_low/g_1_5_6pack',
                    44: 'morshin_b/g_1_5_6pack',
                    45: 'morshin_3_2pack',
                    46: 'truskavetska_b/g_05',
                    47: 'borjomi_6pack',
                    48: 'morshin_low/g_12pack',
                    49: 'morshin_b/g_033_12pack',
                    50: 'morshin_b/g_12pack',
                    51: 'morshin_premium_b/g_12pack',
                    52: 'morshin_premium_low/g_12pack',
                    53: 'morshin_premium_b/g_6pack',
                    54: 'morshin_premium_low/g_6pack',
                    55: 'pack_spiderman',
                    56: 'pack_peppa_pig',
                    57: 'morshin_sport_12pack',
                    58: 'mirgorod_05_12pack',
                    59: 'pack_pine_water_lemon',
                    60: 'pack_ pine_water_ bilberry',
                    61: 'pack_ pine_water_mint',
                    62: 'pack_sportic',
                    63: 'pack_Junior',
                    64: 'pack_truskav_b/g_05',
                    65: 'pack_truskav_low/g_05',
                    66: 'pack_truskav_g_05',
                    67: 'pack_truskav_g_1_5',
                    68: 'pack_truskav_b/g_1_5',
                    69: 'pack_truskav_low/g_1_5',
                    70: 'morshin_g_6pack',
                    71: 'pack_aqualife',
                    72: 'pack_morshinka_033',
                    73: 'pack_aquanyanya_sport',
                    74: 'pack_aquanyanya_033',
                    75: 'pack_morshinka_sport',
                    76: 'pack_morshinka_1_5',
                    77: 'aqua_nyanya_1_5_6pack',
                    78: 'borjomi_033_CAN_12pack_summer',
                    79: 'borjomi_12pack',
                    80: 'pack_borjomi_gb',
                    81: 'borjomi_033_CAN_4pack_summer',
                    82: 'borjomi_05_glass_5+1promo_pack',
                    83: 'morshinka_033',
                    84: 'morshin_b/g_05_5+1promo_pack',
                    85: 'morshin_sport_5+1promo_pack',
                    86: 'morshin_6_2pack',
                    87: 'pack_princess',
                    88: 'borjomi_033_CAN_4pack_regular',
                    89: 'morshin_b/g_1_5_5+1promo_pack',
                    90: 'morshin_low/g_1_5_5+1promo_pack',
                    91: 'truskavetska_low/g_05',
                    92: 'borjomi_g/b_summer',
                    93: 'morshin_033_Junior',
                    94: 'morshin_b/g_1_5_12pack',
                    95: 'borjomi_4pack_summer',
                    96: 'morshin_b/g_05_6pack',
                    97: 'pack_morshinka_sportik',
                    98: 'mirgorod_lagidna_05_12pack',
                    99: 'Borjomi 0.33 CAN 4 pack summer',
                    100: 'Borjomi_0.33_CAN_4_ pack_summer'
                   }

img_test = 'images/Z1_outsfile0img2_106.jpg'
boxes_TTA,scores_TTA,labels_TTA = TTA(model, img_test)


draw_ = read_image_bgr(img_test)
draw = cv2.cvtColor(draw_, cv2.COLOR_BGR2RGB)
for box, score, label in zip(boxes_TTA[0], scores_TTA[0], labels_TTA[0]):


    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(25, 25))
plt.axis('off')
plt.imshow(draw)
plt.savefig('test.jpg')