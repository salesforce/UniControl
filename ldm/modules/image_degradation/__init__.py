'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
'''

from ldm.modules.image_degradation.bsrgan import degradation_bsrgan_variant as degradation_fn_bsr
from ldm.modules.image_degradation.bsrgan_light import degradation_bsrgan_variant as degradation_fn_bsr_light
