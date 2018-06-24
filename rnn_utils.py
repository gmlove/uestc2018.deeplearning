from io import open
import random
import torch

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    import re
    def normalizeEngString(s):
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def normalizeChnString(s):
        s = " ".join(s)
        return s

    print("Reading lines...")

    # Read the file and split into lines
    lines = open('datasets/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    normalizers = {'cmn': normalizeChnString, 'eng': normalizeEngString}
    langs = [lang1, lang2]
    normalize = lambda i, s: normalizers[langs[i]](s)
    pairs = [[normalize(i, s) for i, s in enumerate(l.split('\t'))] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



def prepareData(lang1, lang2, max_length, device, reverse=False):
    def filterPair(p):
        return len(p[0].split(' ')) < max_length and \
            len(p[1].split(' ')) < max_length

    def filterPairs(pairs):
        return [pair for pair in pairs if filterPair(pair)]

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return Dataset(input_lang, output_lang, pairs, device)


class Dataset():
    
    def __init__(self, input_lang, output_lang, pairs, device):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.device = device
    
    def randomPair(self):
        return random.choice(self.pairs)
    
    def random(self):
        return self.toTensor(self.randomPair())
    
    def sentenceToTensor(self, lang, sentence):
        def _indexesFromSentence(lang, sentence):
            return [lang.word2index[word] for word in sentence.split(' ')]
        indexes = _indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def wordFromOutputLang(self, idx):
        return self.output_lang.index2word[idx]
    
    def sentenceToTensorFromInputLang(self, sentence):
        return self.sentenceToTensor(self.input_lang, sentence)
    
    def toTensor(self, pair):
        input_tensor = self.sentenceToTensor(self.input_lang, pair[0])
        target_tensor = self.sentenceToTensor(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    
import time
import math


def _asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (_asMinutes(s), _asMinutes(rs))