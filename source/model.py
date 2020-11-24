from depen import *
from modules import Model_Check


class Multi_Synth_pl(pl.LightningModule):
    def __init__(self, hparams, steps_per_epoch):
        super().__init__()
        self.hparams = hparams
        self.network = Model_Check(self.hparams)
        self.save_hyperparameters('steps_per_epoch')
        self.loss = nn.MSELoss()

    def forward(self, text_input, text_mask, audio_input, audio_mask):
        return self.network(text_input, text_mask, audio_input, audio_mask)
    
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
    
    
    def batch_doing(self, batch):
        sentences_tensor, sentences_mask, spectrograms, mel_mask, waveforms, waveform_l, client_ids, example_ids = batch
        if self.hparams.mask_reverse:
            sentences_mask = sentences_mask == False
            mel_mask = mel_mask == False
        
        if self.hparams.separate_example:
            repetition_per_example = int(len(spectrograms)/len(example_ids)) - 1
            new_indecies = np.delete(np.arange(len(spectrograms)), example_ids)

            spectrograms_examples = torch.repeat_interleave(spectrograms[example_ids], repeats = repetition_per_example, dim=0)
            mel_mask_examples = torch.repeat_interleave(mel_mask[example_ids], repeats = repetition_per_example, dim=0)

            sentences_tensor = sentences_tensor[new_indecies]
            sentences_mask = sentences_mask[new_indecies]
            spectrograms = spectrograms[new_indecies]
        else:
            mel_mask_examples = mel_mask
            spectrograms_examples = torch.clone(spectrograms)
        #mel_mask = spectrograms[new_indecies]

        return sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples, spectrograms

    
    def training_step(self, batch, batch_idx):
        sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples, spectrograms = self.batch_doing(batch)

        x = self(sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples)
        x = x.unsqueeze(1).transpose(2,3).contiguous()
        loss = self.loss(x, spectrograms)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def training_epoch_end(self, outputs):
    #    avg_loss = torch.stack(outputs).mean()
        self.log('epoch_now', self.current_epoch, logger=True)
    #    return {'avg_train_loss': avg_loss}
    #def training_step_end(self, outputs):
    #    avg_loss = outputs.mean()
    #    return {'avg_train_loss': avg_loss}
    
    def validation_step(self, batch, batch_idx):
        sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples, spectrograms = self.batch_doing(batch)

        x = self(sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples)
        x = x.unsqueeze(1).transpose(2,3).contiguous()
        loss = self.loss(x, spectrograms)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #def validation_epoch_end(self, outputs):
    #    avg_loss = outputs.mean()
    #    return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples, spectrograms = self.batch_doing(batch)
        #([20, 1, 128, 1602])
        x = self(sentences_tensor, sentences_mask, spectrograms_examples, mel_mask_examples)
        x = x.unsqueeze(1).transpose(2,3).contiguous()
        loss = self.loss(x, spectrograms)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #def test_epoch_end(self, outputs):
    #    avg_loss = outputs.mean()
    #    self.log('avg_test_loss', avg_loss, on_epoch=True, logger=True)
    #    return {'avg_test_loss': avg_loss}
