INPUT_W = 512
INPUT_H = 64
TRAIN_TXT = './dataset/Train/NDtrain.txt'
AEVAL_TXT = './dataset/Train/NDeval.txt'
ALPHA = 0.2
EPOCH = 100
BATCH_SIZE = 20
PEOPLE_PER_BATCH = 25
IMGS_PER_PERSON = 40
LR = 1e-3
LR_DECAY = 1e-6

from model import UniNet

if __name__ == '__main__':
    uninet = UniNet(
        input_w=INPUT_W,
        input_h=INPUT_H,
        train_txt=TRAIN_TXT,
        alpha=ALPHA,
        epoch=EPOCH,
        batch_size=BATCH_SIZE,
        people_per_batch=PEOPLE_PER_BATCH,
        imgs_per_person=IMGS_PER_PERSON,
        lr=LR,
        lr_decay=LR_DECAY
    )

    uninet.train()