import numpy as np
import torch
from collections import defaultdict
import json

def xor(n):
    x = torch.rand(n, 2)
    c = (x > 0.5)*1.0
    y = c.sum(1)%2
    y = y.unsqueeze(1)
    return c, y

def three_xor(n):
    x = torch.rand(n, 4)
    c = (x > 0.5)*1.0
    y = c.sum(1)%2
    y = y.unsqueeze(1)
    return c, y

def clevr_logic(n):
    x = torch.rand(n, 4)
    c = (x > 0.5)*1.0
    y1 = (c[:, 0] + c[:, 1])%2
    y2 = (1-y1)*c[:, 2]
    y3 = (1-y1)*(1-c[:, 2])
    
    y = torch.stack([y1, y2, y3], dim=1)
    _, idx = y.max(dim=1)
    y = idx
    y = y.unsqueeze(1)
    
    return c, y
        
def create_clevr_logic_dicts():
    shapes = np.array(["cube", "sphere", "cylinder", "cone"])
    
    x = torch.rand(5000, 4)
    c = (x > 0.5)*1.0
    y1 = (c[:, 0] + c[:, 1])%2
    y2 = (1-y1)*c[:, 2]
    y3 = (1-y1)*(1-c[:, 2])
    
    y = torch.stack([y1, y2, y3], dim=1)
    _, idx = y.max(dim=1)
    y = idx
    y = y.unsqueeze(1)
    
    cls1_indices, cls2_indices, cls3_indices,  = [], [], []
    for i in range(y.shape[0]):
        if y1[i] == 1:
            cls1_indices.append(i)
        elif y2[i] == 1:
            cls2_indices.append(i)
        elif y3[i] == 1:
            cls3_indices.append(i)

    dicto1, dicto2, dicto3  = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(c.shape[0]):
        shapo = []
        for j in range(c.shape[1]):
            if c[i, j]:
                shapo.append(shapes[j])
        if not shapo:
            continue
        if i in cls1_indices:
            dicto1['image' + str(i)] = shapo
        if i in cls2_indices:
            dicto2['image' + str(i)] = shapo
        if i in cls3_indices:
            dicto3['image' + str(i)] = shapo
    
    with open("dicto1.json", "w") as file:
        json.dump(dicto1, file, indent=4)
    with open("dicto2.json", "w") as file:
        json.dump(dicto2, file, indent=4)
    with open("dicto3.json", "w") as file:
        json.dump(dicto3, file, indent=4)