#fast ai version1 keras - lesson1 on keras 2

path = "D:\\datasets\\dogsvscatssmall\\sample\\"

batch_size=4

import vgg16n;
from vgg16n import Vgg16n

vgg = Vgg16n()

train_generator = vgg.get_generator(path+'train', batch_size=batch_size)
val_generator = vgg.get_generator(path+'valid', batch_size=batch_size)

vgg.finetune(train_generator)
vgg.fitvaldata(train_generator, val_generator, nb_epoch=1, batch_size=batch_size)