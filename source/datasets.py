from depen import *
from depen import tokenizer, vocab, token_indexer
import phonemizer
from phonemizer import phonemize
from phonemizer import separator
from dataset_download import download_dataset_and_extract


def clean(tsvm, mel_limit):
  df = pd.read_csv(tsvm, sep="\t")
  listt = df[["mel_l", "wav_l"]]

  listt = listt.sort_values("mel_l")
  print(np.mean(listt["mel_l"].values))
  cleaned = listt.loc[listt['mel_l'] < mel_limit]
  indexx = cleaned.index
  #cleaned_list = listt.loc[indexx]
  print(df)
  df = df.loc[indexx].sort_index()
  print(df)
  return df

def tsv_common_reconstruction(path, TSV, train_size = 50, n_limit = 500, upper_limit = 1000, mel_limit = 2500):

  n_limit = n_limit - 1
  mel_limit = mel_limit - 1
  tsvm = os.path.join(path, TSV)
  df = clean(tsvm, mel_limit)

  #print("unlimited matrix")
  #print(df)

  if "up_votes" in df:
    df = df.drop(columns=['up_votes', 'down_votes', 'age', "gender", 	"accent", 	"locale", 	"segment"])


  #create column of len, then delete

  df_unique = df['client_id'].value_counts()
  limit = np.sum(df_unique.values > n_limit)

  names = df_unique.head(limit)
  names = names.axes[0].tolist()

  assert len(names) == limit

  index_range = list(range(0, limit))

  df = df.loc[df['client_id'].isin(names)] #cutted
  df["client_id"] = df["client_id"].replace(names, index_range)
  
  #print("original matrix")
  #print(df)
  
  
  numpy_array = df.values

  ids, indices, counts = np.unique(numpy_array[:, 0], return_index=True, return_counts=True)

  #print("original counts")
  #print(ids, "\n", indices, "\n", counts)
  
  inds = indices.argsort()
  new_indices = indices[inds]
  new_counts = counts[inds]
  
  new_counts = new_counts - new_counts%train_size

  new_counts[new_counts>upper_limit] = upper_limit

  new_idx = []

  for i in range(len(new_counts)):
    ranges = list(range(new_indices[i], new_indices[i] + new_counts[i]))
    new_idx = new_idx + ranges
  
  arr = numpy_array[new_idx]

  ids, indices, counts = np.unique(arr[:, 0], return_index=True, return_counts=True)

  #print("new counts")
  #print(ids, "\n", indices, "\n", counts)

  df = pd.DataFrame(arr, columns=df.columns)

  print("new df")
  print(df)
  
  lent = df.values.shape[0]
  print(lent)
  new_TSV = "end_" + TSV
  tsvm = os.path.join(path, new_TSV)
  df.to_csv(tsvm, sep = '\t', index=False)
  return limit, lent



# copied from official pytorch docs
def load_commonvoice_item(line: List[str],
                          header: List[str],
                          path: str,
                          folder_audio: str,
                          dias_ph = False) -> Tuple[Tensor, int, Dict[str, str]]:
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent
  
  assert header[1] == "path"
  fileid = line[1]
  client_id = line[0]

  if dias_ph:
    sentence = line[5]
  else:
    sentence = line[2]


  filename = os.path.join(path, folder_audio, fileid)
  waveform, sample_rate = torchaudio.load(filename)
  return waveform, sample_rate, client_id, sentence


# copied from official pytorch docs
class COMMONVOICE(Dataset):
  _ext_txt = ".txt"
  _ext_audio = ".mp3"
  _folder_audio = "clips"

  def __init__(self,
                 root: str,
                 tsv: str,
                 multi_size, # 10
                 mel_limit,
                 multinumber = True,
                 reconstructed = False,
                 dias_ph = False) -> None:
  
    #if dias_ph:
    #  assert dias_ph == reconstructed
    self.dias_ph = dias_ph
    self.multinumber = multinumber
    self.multi_size = multi_size

    #train_size = 10, n_limit = 500, upper_limit = 1000
    if multinumber:
      if tsv[-9:] == "train.tsv":
        n_limit = 500
        upper_limit = 1000
        print("train set")
      else:
        n_limit = 2
        upper_limit = 100000
        print("test set multinumber")
    
    self._path = root

    if self.multinumber:
      if not reconstructed:
        self.limit, self.len_got = tsv_common_reconstruction(self._path, tsv, self.multi_size, n_limit, upper_limit, mel_limit=mel_limit)
        tsv = "end_" + tsv
      else:
        tsv = tsv
    self._tsv = os.path.join(self._path, tsv)

    with open(self._tsv, "r") as tsv:
        walker = unicode_csv_reader(tsv, delimiter="\t")
        self._header = next(walker)
        self._walker = list(walker)
    
  def __getitem__(self, n: int):

    if not self.multinumber:
      line = self._walker[n]
      return load_commonvoice_item(line, self._header, self._path, self._folder_audio, self.dias_ph), False
    else:
      new_n = n * self.multi_size
      sample_rates = []
      client_ids = []
      sentences = []
      sentences_l = []
      waveforms = []
      waveform_l = []
      
      for i in range(self.multi_size):
        line = self._walker[new_n + i]
        waveform, sample_rate, client_id, sentence = load_commonvoice_item(line, self._header, self._path, self._folder_audio, self.dias_ph)
        waveforms.append(waveform)
        waveform_l.append(waveform.shape[1])
        sample_rates.append(sample_rate)
        client_ids.append(client_id)
        sentences.append(sentence)
        sentences_l.append(len(sentence))
      
      outputs = (waveforms, waveform_l, sample_rates, client_ids, sentences, sentences_l)
      return outputs, True
  
  def __len__(self) -> int:
    if self.multinumber:
      return int( len(self._walker) / self.multi_size)
    else:
      return len(self._walker)


def create_mask_pad(max_len, list_of_l):
  true_list = np.full((len(list_of_l), max_len), True)
  for i in range(len(list_of_l)):
    aranged = np.arange(list_of_l[i])
    true_list[i][aranged] = False
  return torch.from_numpy(true_list)


def collate_fn_common(batch, data_type, max_len_mel = 2000, reconstructed_phoneme = False):
  #
  ggg, truefalse = batch[0]
  batch_size = len(batch)
  parts = 6
  final_list = []

  if truefalse:
    seq_len = len(ggg[1])
    del ggg
    example_id = []
    for n in range(parts):
      final_list.append([None]*(batch_size*seq_len))
    #
    for i in range(batch_size):
      part, truefalse = batch[i]
      for j in range(parts):
        final_list[j][i*seq_len:(i+1)*seq_len] = part[j]
      example_id.append(i*seq_len)
    #
  else:
    del ggg
    for n in range(parts):
      final_list.append([None]*(batch_size))
    #
    for i in range(batch_size):
      part, truefalse = batch[i]
      waveform, sample_rate, client_id, sentence = part
      part = [waveform, waveform.shape[1], sample_rate, client_id, sentence, len(sentence)]
      for j in range(parts):
        final_list[j][i] = part[j]
    example_id = None
  #
  waveforms, waveform_l, sample_rates, client_ids, sentences, sentences_l = final_list
  #
  if not reconstructed_phoneme:
    sentences = phonemize(sentences, backend='espeak',with_stress=False, separator=separator.Separator(phone=' ', syllable='', word='- '))
  for i in range(len(sentences_l)):
    sentences_l[i] = len(sentences[i])
  biggest_l_index = sentences_l.index(max(sentences_l))
  
  token = tokenizer.tokenize(sentences[biggest_l_index])
  text_field = TextField(token, token_indexer)
  text_field.index(vocab)
  padding_lengths = text_field.get_padding_lengths()

  list_tokens = []
  mel_list = [None] * len(sample_rates)
  mel_list_l = [None] * len(sample_rates)
  #

  for i in range(len(sentences_l)):
    token = tokenizer.tokenize(sentences[i])
    text_field = TextField(token, token_indexer)
    text_field.index(vocab)
    tensor_dict = text_field.as_tensor(padding_lengths)
    list_tokens.append(tensor_dict)
    #
    if data_type == "train":
      mel_list[i] = train_audio_transforms(waveforms[i]).squeeze(0).transpose(0, 1)
    else:
      mel_list[i] = test_audio_transforms(waveforms[i]).squeeze(0).transpose(0, 1)
    mel_list_l[i] = mel_list[i].shape[0]
    waveforms[i] = waveforms[i].squeeze(0)
    #
  waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)
  #
  mel_list.append(torch.zeros((max_len_mel, mel_list[0].shape[1])))
  spectrograms = nn.utils.rnn.pad_sequence(mel_list, batch_first=True).unsqueeze(1).transpose(2, 3)
  spectrograms = spectrograms[1:]
  highest_mel_l = spectrograms[0].shape[2]
  mel_mask = create_mask_pad(highest_mel_l, mel_list_l)
  #
  text_field_tensors = text_field.batch_tensors(list_tokens)
  #
  sentences_tensor = nn_util.get_token_ids_from_text_field_tensors(text_field_tensors)
  sentences_mask = nn_util.get_text_field_mask(text_field_tensors) == False

  return sentences_tensor, sentences_mask, spectrograms, mel_mask, waveforms, waveform_l, client_ids, example_id


class Common_pl_dataset(pl.LightningDataModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.path = self.hparams.path_dataset_common
    if not self.hparams.reconstructed_phoneme:
      print("!!!!!!!!!!!!!!!!!!!!!!!!!! prepare long time !!!!!!!!!!!!!!!!!!!!")


  def prepare_data(self):
    #here you download your dataset, from some site for example
    #or pytorch torchaudio etc.
    #or call torch.utils.data.Dataset type
    download_dataset_and_extract("COMMONVOICE")

  def setup(self): 
    TSV_train, TSV_test, TSV_val = self.hparams.train_path, self.hparams.dev_path, self.hparams.test_path
    self.dataset_train = COMMONVOICE(self.hparams.path_dataset_common, TSV_train, self.hparams.drow_train_examples, mel_limit=self.hparams.mel_limit, reconstructed = self.hparams.reconstructed, dias_ph = self.hparams.dias_ph)
    self.dataset_test = COMMONVOICE(self.hparams.path_dataset_common, TSV_test, 2, mel_limit=self.hparams.mel_limit, reconstructed =self.hparams.reconstructed, dias_ph = self.hparams.dias_ph)
    self.dataset_val = COMMONVOICE(self.hparams.path_dataset_common, TSV_val, 2, mel_limit=self.hparams.mel_limit, reconstructed = self.hparams.reconstructed, dias_ph = self.hparams.dias_ph)

  def train_dataloader(self):
    train_loader = DataLoader(dataset=self.dataset_train,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                collate_fn=lambda x: collate_fn_common(x, 'train', self.hparams.mel_limit, reconstructed_phoneme = self.hparams.reconstructed_phoneme),
                                num_workers = self.hparams.num_workers)
    return train_loader

  def val_dataloader(self):
    val_loader = DataLoader(dataset=self.dataset_val,
                                batch_size=self.hparams.batch_size * int(self.hparams.drow_train_examples / 2),
                                shuffle=False,
                                collate_fn=lambda x: collate_fn_common(x, 'valid', self.hparams.mel_limit, reconstructed_phoneme = self.hparams.reconstructed_phoneme),
                                num_workers = self.hparams.num_workers)
    return val_loader

  def test_dataloader(self):
    test_loader = DataLoader(dataset=self.dataset_test,
                                batch_size=self.hparams.batch_size * int(self.hparams.drow_train_examples / 2),
                                shuffle=False,
                                collate_fn=lambda x: collate_fn_common(x, 'valid', self.hparams.mel_limit, reconstructed_phoneme = self.hparams.reconstructed_phoneme),
                                num_workers = self.hparams.num_workers)
    return test_loader
