from depen import *
from modules import Model_dl


def _accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions

    parameters -------------------------
    - output        -   model output tensor
    - target        -   actual label tensor
    - topk          -   top k accuracy values to return

    returns ----------------------------
    - res           -   list of k top accuracies
    """

    num_classes = 1
    for dim in output.shape[1:]:
        num_classes *= dim

    with torch.no_grad():
        maxk = max(topk)
        maxk = min(maxk, num_classes)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k < num_classes:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append([0, 0])

        return res


class Age_Gender_pl(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
        super().__init__()
        self.save_hyperparameters(conf)
        self.save_hyperparameters()
        print(self.hparams)
        #self.hparams = hparams
        self.network = Model_dl(self.hparams)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, image):
        return self.network(image)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
	                                                                        max_lr=self.hparams.learning_rate,
	                                                                        steps_per_epoch=self.hparams.steps_per_epoch, #int(len(train_loader))
	                                                                        epochs=self.hparams.epochs,
	                                                                        anneal_strategy='linear'),
                        'name': 'lr_scheduler_lr',
                        'interval': 'step', # or 'epoch'
                        'frequency': 1,
                        }
        if self.hparams.add_sch:
            return [optimizer], [lr_scheduler]
        else:
            return optimizer
    

    def count_accuracy(self, y_got, labels):
        acc = _accuracy(y_got, labels)
        return acc[0]

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        y_got = self(images)
        loss = self.loss(y_got, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = self.count_accuracy(y_got, labels)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def training_epoch_end(self, outputs):
        self.log('epoch_now', self.current_epoch, logger=True)

    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        y_got = self(images)
        loss = self.loss(y_got, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = self.count_accuracy(y_got, labels)
        self.log('vall_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #def validation_epoch_end(self, outputs):
    #    avg_loss = outputs.mean()
    #    return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        y_got = self(images)
        loss = self.loss(y_got, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = self.count_accuracy(y_got, labels)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #def test_epoch_end(self, outputs):
    #    avg_loss = outputs.mean()
    #    self.log('avg_test_loss', avg_loss, on_epoch=True, logger=True)
    #    return {'avg_test_loss': avg_loss}
