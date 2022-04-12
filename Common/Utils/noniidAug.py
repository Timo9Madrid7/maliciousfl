import torchvision
import random

def mnist_noniid_aug(data, rotation_range=30, shift_range=0.2, shear_range=45, pad_val=4):
    slots = [[] for _ in range(10)]
    total_num_samples = len(data)
    for sample in data:
        slots[sample[1]].append(sample[0])
    major_num_samples, major_class = 0, 0
    for i in range(10):
        if major_num_samples < len(slots[i]):
            major_num_samples = len(slots[i])
            major_class = i
    q = major_num_samples/total_num_samples  
    if q <= 0.2:
        return data

    minor_classes = list(range(10))
    minor_classes.pop(major_class)
    noniid_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(degrees=rotation_range,
                                            translate=(shift_range, shift_range),
                                            shear=shear_range
                                        ),
        torchvision.transforms.RandomCrop(28, pad_val),
    ])
    
    new_data = list(zip(slots[major_class], [major_class]*major_num_samples))
    for i in minor_classes:
        minor_num_samples = len(slots[i])
        if minor_num_samples > 0:
            chosen_samples = random.choices(slots[i], k=int((4/(1/q-1)-1)*minor_num_samples))
            slots[i] += list(map(noniid_aug, chosen_samples))
            new_data += list(zip(slots[i], [i]*len(slots[i])))

    return new_data