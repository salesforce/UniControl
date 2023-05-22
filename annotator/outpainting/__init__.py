'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu
'''

import numpy as np

class Outpainter:
    def __call__(self, img, rand_h, rand_w):
        h = img.shape[0]
        w = img.shape[1]
        h_new = int(float(h) / 100.0 * float(rand_h))
        w_new = int(float(w) / 100.0 * float(rand_w))
        img_new = np.zeros(img.shape)
        img_new[(h-h_new)//2:(h+h_new)//2, (w-w_new)//2:(w+w_new)//2] = img[(h-h_new)//2:(h+h_new)//2, (w-w_new)//2:(w+w_new)//2]
        img_new = img_new.astype('ubyte')
        return img_new
