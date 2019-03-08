import torch
from torch.autograd import Variable
from crnn_pytorch import utils
from crnn_pytorch import dataset
from PIL import Image
import PIL

from crnn_pytorch.models import crnn



def img_read(img):
    model_path = './crnn_pytorch/crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    # print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    #image = Image.open(img).convert('L')
    image = PIL.Image.fromarray(img[:,:,0],'L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred
