from depen import *

class AdienceDataset(Dataset) :
    def __init__(self, txt_files, root_dir, transform) :
        self.txt_files = txt_files
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.read_from_txt_files()

    def __len__(self) :
        return len(self.data)

    def read_from_txt_file(self,txt_file) :
        data = []
        f = open(txt_file)
        for line in f.readlines() :
            image, label = line.split()
            label = int(label)
            data.append((image, label))
        return data

    def read_from_txt_files(self):
        for txt_file in self.txt_files:
            self.data.extend(self.read_from_txt_file(txt_file))


    def __getitem__(self, idx) :
        img, label = self.data[idx]
        image = Image.open(os.path.join(self.root_dir, img))

        if self.transform :
            image = self.transform(image)

        return {
            'image' : image,
            'label' : label
        }

def get_dataloader(path_to_data, image_size, types, category, fold_numbers, transform_index, minibatch_size, num_workers, Gray=False) :
    PATH_TO_FOLDS = "train_val_txt_files_per_fold"
    PATH_TO_IMAGE_FOLDERS = os.path.join(path_to_data, "aligned")
    applied_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomCrop(227),
        transforms.Grayscale(),
        transforms.GaussianBlur(kernel_size=(65, 65)),
        transforms.RandomRotation(20)
    ]

    dictionary_of_transformations = {
        'train' : {
            0 : list(applied_transforms[i] for i in [0, 1, 3]),  # no transformation
            1 : list(applied_transforms[i] for i in [0, 1, 2, 3]),  # random horizontal flip
            2 : list(applied_transforms[i] for i in [0, 4, 2, 3]),  # random crop and random horizontal flip
            3 : list(applied_transforms[i] for i in [0, 1, 6, 3]),
            4 : list(applied_transforms[i] for i in [0, 1, 7, 3]),
            5: list(applied_transforms[i] for i in [0, 1, 2, 5, 3])
        },
        'val' : {
            0 : list(applied_transforms[i] for i in [0, 1, 3]),
            5:  list(applied_transforms[i] for i in [0, 1, 5, 3])
        },
        'test' : {
            0 : list(applied_transforms[i] for i in [0, 1, 3]),
            5:  list(applied_transforms[i] for i in [0, 1, 5, 3])
        }
    }
    if Gray:
        transform_index = 5
    #s test, val, train
    #c age, gender
    #fold 0-4
    #transform index corresponding to train,test,val
    txt_files = []
    for fold_number in fold_numbers:
        test_fold = 'test_fold_is_' + str(fold_number) + '/' +category + '_' +types+'.txt'
        txt_file = os.path.join(PATH_TO_FOLDS,test_fold)
        txt_files.append(txt_file)
    root_dir = PATH_TO_IMAGE_FOLDERS
    if types == "val" or types == "test":
        shuffle = False
    else:
        shuffle = True
    transformed_dataset = AdienceDataset(txt_files, root_dir, 
                                         transforms.Compose(dictionary_of_transformations[types][transform_index]))
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader




class Adience_pl_dataset(pl.LightningDataModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.path = self.hparams.path_to_files


  def prepare_data(self):
    #here you download your dataset, from some site for example
    #or pytorch torchaudio etc.
    #or call torch.utils.data.Dataset type
    print("already prepared but later add")

  def setup(self): 
    print("ready")

  def train_dataloader(self):
    train_loader = get_dataloader(path_to_data=self.path, image_size = self.hparams.im_size, types = "train",
                                  category = self.hparams.category, fold_numbers = [self.hparams.fold_numbers],
                                  transform_index = 1, minibatch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers,
                                  Gray=self.hparams.grayscale)
    return train_loader

  def val_dataloader(self):
    val_loader = get_dataloader(path_to_data=self.path, image_size = self.hparams.im_size, types = "val",
                                  category = self.hparams.category, fold_numbers = [self.hparams.fold_numbers],
                                  transform_index = 0, minibatch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers,
                                  Gray=self.hparams.grayscale)
    return val_loader

  def test_dataloader(self):
    test_loader = get_dataloader(path_to_data=self.path, image_size = self.hparams.im_size, types = "test",
                                  category = self.hparams.category, fold_numbers = [self.hparams.fold_numbers],
                                  transform_index = 0, minibatch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers,
                                  Gray=self.hparams.grayscale)
    return test_loader
