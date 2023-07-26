'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin and Shu Zhang
'''


from safetensors.torch import load_file as stload
from collections import OrderedDict
from cldm.model import create_model

st_path = '../ckpts/unicontrol_v1.1.st'
model_dict = OrderedDict(stload(st_path, device='cpu'))

model = create_model('./models/cldm_v15_unicontrol.yaml').cpu()
model.load_state_dict(model_dict, strict=False)
model = model.cuda()
