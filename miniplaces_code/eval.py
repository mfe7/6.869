import gc
import sys
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

dir_path = os.path.dirname(os.path.realpath(__file__))

def run():
    # Parameters
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model.load_state_dict(torch.load("models/model.10"))
    model = model.to(device)

    test_loader = dataset.get_test_loader(batch_size)
    num_test_batches = len(test_loader)

    model.eval()
    eval_filename = "{dir}/results.txt".format(dir=dir_path)
    with torch.no_grad():
        with open(eval_filename, "w") as file:
            img_id = 0
            for batch_num, (inputs, labels) in enumerate(test_loader, 1):
                inputs = inputs.to(device)
                outputs = model(inputs)
                top5 = outputs.topk(5)[1]

                for i in range(inputs.size()[0]):
                    img_id += 1
                    file.write("test/{name}.jpg {first} {second} {third} {fourth} {fifth}\n".format(name=str(img_id).zfill(8), first=top5[i,0], second=top5[i,1], third=top5[i,2], fourth=top5[i,3], fifth=top5[i,4]))

                print(batch_num*1.0/num_test_batches)
                # torch.save(model.state_dict(), "models/model.%d" % epoch)

print('Starting evaluation...')
run()
print('Evaluation finished.')
