import os
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import ClassificationModel
from .utils import Progbar, create_dir

class Classifier():
    def __init__(self, config,name):
        self.config = config

        model_name = name

        self.debug = False
        self.model_name = model_name
        self.classification_model = ClassificationModel(model_name,'classification_model',config).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset =Dataset(config,config.ori_val, augment=False, training=True)
        else:
            self.train_dataset = Dataset(config, config.ori_train, augment=True, training=True)
            self.val_dataset = Dataset(config, config.ori_val, augment=False, training=True)
        self.sampler=Dataset(config, config.ori_val,augment=False, training=True)
        self.results_path = os.path.join(config.PATH, 'results')


        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True
        create_dir(os.path.join(config.PATH,model_name))
        self.log_file = os.path.join(os.path.join(config.PATH,model_name), 'log_' + model_name + '.dat')

    def load(self):
        self.classification_model.load()

    def save(self):
        self.classification_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.classification_model.train()

                images,labels= self.cuda(*items)


                outputs, loss, logs,precision = self.classification_model.process(images,labels)
                #print(outputs)


                # backward
                self.classification_model.backward(loss)
                iteration = self.classification_model.iteration





                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs
                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints


                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.classification_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, labels = self.cuda(*items)

            outputs, loss, logs, precision = self.classification_model.process(images, labels)



            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)



    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


    def process(self,img):

        with torch.no_grad():
            img=self.sampler.generate_test_data(img)
            outputs = self.classification_model(img)
            outputs=outputs.cpu().numpy()
            #print(outputs)
            return outputs
