import os
import random
import torch
import numpy
import time
from torch.utils.data import Dataset
import argparse
import copy


# define veribles
BATCH_SIZE = 20
# NUM_EPOCHS = 20
LEARNING_RATE = 0.05
# FILE_NAME = "./input/onegin.txt"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # choice device
# CAESAR_OFFSET = 10

# RNN class
class RNNModel(torch.nn.Module):

    def __init__(self, size_of_dict=200, embedd_dim=200, hidden_size=200, out_size=200):
        super(RNNModel, self).__init__()
        self.embed = torch.nn.Embedding(size_of_dict, embedd_dim)
        self.rnn = torch.nn.RNN(embedd_dim, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, out_size)

    def forward(self, sentence):
        embed = self.embed(sentence)
        o, h = self.rnn(embed)
        return self.linear(o)


# function for load alphabet from traning text
def load_alphabet(file):
    alphabet = list(set(file))
    alphabet.sort()
    for ch in ['\t', '\n', ' ']:
        if ch in alphabet:
            alphabet.remove(ch)

    return alphabet


# function for loading text
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# function for save text
def save_text(text, name):
    if not os.path.exists('./output'):
        os.mkdir('./output')
    with open(f'./output/{name}.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(text))


# caesar cipher function
def caesar_cipher(text, alphabet, key=1):
    res, n = [], ""
    text = text.split(' ')
    key = key if (key >= 0 & key < len(alphabet)) else (key % len(alphabet))
    for word in text:
        lst = []
        for ch in word:
            if ch in alphabet:
                idx = (alphabet.index(ch) + key) % len(alphabet)
                lst.append(alphabet[idx])
            else:
                lst.append(ch)
        res.append(''.join(lst))

    return text, res


#
def dict_function(chars):
    index_to_char = ['none'] + [w for w in chars]
    char_to_index = {w: i for i, w in enumerate(index_to_char)}
    return index_to_char, char_to_index


# function to convert text to tensor
def char_to_tensor(text, char_to_index):
    MAX_LEN = max(len(x) for x in text)
    X = torch.zeros((len(text), MAX_LEN), dtype=int)

    for i in range(len(text)):
        for j, w in enumerate(text[i]):
            X[i, j] = char_to_index.get(w, char_to_index['none'])
    return X


# function to convert tensor to text
def tensor_to_text(code, index_to_char):
    out = []
    for word in code:
        lst = []
        for i in word:
            i %= len(index_to_char)
            if index_to_char[i] != 'none':
                lst.append(index_to_char[i])
        out.append(''.join(lst))
    return out


# function for shuffling and splitting into samples
def shuffle_data(X, Y, test_size=0.2, val_size=0.1):
    lst = [i for i in range(X.shape[0])]
    # random.shuffle(lst)
    val_lst = lst[: int(len(lst) * val_size)]
    lst = lst[int(len(lst) * val_size):]
    X_val = X[val_lst, :]
    Y_val = Y[val_lst, :]
    X, Y = X[lst, :], Y[lst, :]
    lst = [i for i in range(X.shape[0])]
    random.shuffle(lst)
    test_lst = lst[: int(len(lst) * test_size)]
    lst = lst[int(len(lst) * test_size):]
    X_test = X[test_lst, :]
    Y_test = Y[test_lst, :]
    X_train, Y_train = X[lst, :], Y[lst, :]
    return X_train, Y_train, X_test, Y_test, X_val, Y_val



def train(NUM_EPOCHS=10, test_size=0.2, val_size=0.1, FILE_NAME="./input/onegin.txt", hidden_size=150,
          out_size=100, embedd_dim=150, CAESAR_OFFSET=2):
    # load text from path
    print(f'Please wait. Load text...')
    original_text = load_text(FILE_NAME)
    # create alphabet from text
    alphabet = load_alphabet(original_text)
    # create caesar text from original text
    print(f'The text is encoded in a Caesar cipher. Please wait.')
    original_text, caesar_text = caesar_cipher(text=original_text, alphabet=alphabet, key=CAESAR_OFFSET)

    # save text
    save_text(caesar_text, 'caesar')

    index_to_char, char_to_index = dict_function(alphabet)

    # convert our text to tensor
    X = char_to_tensor(text=caesar_text, char_to_index=char_to_index)
    Y = char_to_tensor(text=original_text, char_to_index=char_to_index)

    # samples train, test and val from test
    X_train, Y_train, X_test, Y_test, X_val, Y_val = shuffle_data(X, Y, val_size=val_size, test_size=test_size)
    traindataset = torch.utils.data.TensorDataset(X_train, Y_train)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
    testdataset = torch.utils.data.TensorDataset(X_test, Y_test)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

    # set parameters our model and create model
    size_of_dict = len(char_to_index)
    if embedd_dim < len(char_to_index):
        embedd_dim = size_of_dict
    if out_size < len(char_to_index):
        out_size = size_of_dict
    # hidden_size = 200
    # out_size = 200
    # embedd_dim = 200

    model = RNNModel(size_of_dict=size_of_dict, embedd_dim=embedd_dim, hidden_size=hidden_size,
                     out_size=out_size).to(DEVICE)

    # create loss function and optimizer
    loss = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print(f'Model train')

    # train model
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, iter_num = .0, .0, .0
        start_epoch_time = time.time()
        model.train()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for x_in, y_in in traindataloader:
            x_in = x_in.to(DEVICE)
            y_in = y_in.to(DEVICE)
            y_in = y_in.view(1, -1).squeeze()
            optimizer.zero_grad()
            out = model.forward(x_in).view(-1, out_size).squeeze()
            l = loss(out, y_in)
            train_loss += l.item()
            batch_acc = (out.argmax(dim=1) == y_in)
            train_acc += batch_acc.sum().item() / batch_acc.shape[0]
            l.backward()
            optimizer.step()
            iter_num += 1
        test_loss, test_acc, iter_num = .0, .0, .0
        model.eval()
        for x_in, y_in in testdataloader:
            x_in = x_in.to(DEVICE)
            y_in = y_in.to(DEVICE)
            y_in = y_in.view(1, -1).squeeze()
            out = model.forward(x_in).view(-1, out_size)
            l = loss(out, y_in)
            test_loss += l.item()
            batch_acc = (out.argmax(dim=1) == y_in)
            test_acc += batch_acc.sum().item() / batch_acc.shape[0]
            iter_num += 1
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)


    # check model on validation sample from original text
    val_results = model(X_val.to(DEVICE)).argmax(dim=2)
    val_acc = (val_results == Y_val.to(DEVICE)).flatten()
    val_acc = (val_acc.sum() / val_acc.shape[0]).item()
    out_sentence = tensor_to_text(code=val_results, index_to_char=index_to_char)
    true_sentence = tensor_to_text(code=Y_val, index_to_char=index_to_char)
    print(f"Validation accuracy is : {val_acc:.4f}")
    print("-" * 20)
    print(f"Validation sentence is: \"{' '.join(out_sentence)}\"")
    print("-" * 20)
    print(f"True sentence is:       \"{' '.join(true_sentence)}\"")


    sentence = """Python в русском языке встречаются названия питон или пайтон[24]) —
    высокоуровневый язык программирования общего назначения с динамической строгой типизацией и автоматическим управлением
    памятью[25][26], ориентированный на повышение производительности разработчика, читаемости кода и его качества, а также
    на обеспечение переносимости написанных на нём программ[27]. Язык является полностью объектно-ориентированным в том
    плане, что всё является объектами[25]. Необычной особенностью языка является выделение блоков кода пробельными
    отступами[28]. Синтаксис ядра языка минималистичен, за счёт чего на практике редко возникает необходимость обращаться к
     документации[27]. Сам же язык известен как интерпретируемый и используется в том числе для написания скриптов[25].
     Недостатками языка являются зачастую более низкая скорость работы и более высокое потребление памяти написанных на нём
     программ по сравнению с аналогичным кодом, написанным на компилируемых языках, таких как C или C++[27][25].
     Python является мультипарадигмальным языком программирования, поддерживающим императивное, процедурное, структурное,
     объектно-ориентированное программирование[25], метапрограммирование[29] и функциональное программирование[25].
     Задачи обобщённого программирования решаются за счёт динамической типизации[30][31]. Аспектно-ориентированное
     программирование частично поддерживается через декораторы[32], более полноценная поддержка обеспечивается
     дополнительными фреймворками[33]. Такие методики как контрактное и логическое программирование можно реализовать с
     помощью библиотек или расширений[34]. Основные архитектурные черты — динамическая типизация, автоматическое управление
     памятью[25], полная интроспекция, механизм обработки исключений, поддержка многопоточных вычислений с глобальной
     блокировкой интерпретатора (GIL)[35], высокоуровневые структуры данных. Поддерживается разбиение программ на модули,
     которые, в свою очередь, могут объединяться в пакеты[36]."""


    print("-" * 20)
    original_sent, caesar_sent = caesar_cipher(text=sentence, alphabet=alphabet, key=CAESAR_OFFSET)
    print(f"True sentence is: \"{' '.join(original_sent)}\"")
    print("-" * 20)
    print(f"Caesar sentence is: \"{' '.join(caesar_sent)}\"")

    X_sent = char_to_tensor(text=caesar_sent, char_to_index=char_to_index)
    Y_sent = char_to_tensor(text=original_sent, char_to_index=char_to_index)

    val_results = model(X_sent.to(DEVICE)).argmax(dim=2)
    val_acc = (val_results == Y_sent.to(DEVICE)).flatten()
    val_acc = (val_acc.sum() / val_acc.shape[0]).item()
    out_sentence = tensor_to_text(code=val_results, index_to_char=index_to_char)

    print("-" * 20)
    print(f"Validation sentence is: \"{' '.join(out_sentence)}\"")
    print("-" * 20)
    print(f"Validation accuracy is : {val_acc:.4f}")


parse = argparse.ArgumentParser()
parse.add_argument('--file', '-f', help='Path to file and file name', default="./input/onegin.txt")
parse.add_argument('--epoch', '-e', help='Numbers of epoch', default=20, type=int)
parse.add_argument('--val', '-v', help='Size of validation sample', default=0.1, type=float)
parse.add_argument('--test', '-t', help='Size of test sample', default=0.2, type=float)
parse.add_argument('--hidden', '-hi', help='Number of hidden layer', default=100, type=int)
parse.add_argument('--emb', '-emb', help='Embedded dimensional', default=150, type=int)
parse.add_argument('--outsize', '-o', help='Number of output size', default=150, type=int)
parse.add_argument('--caesar', '-c', help='Caesar offset', default=1, type=int)

args = parse.parse_args()

args = parse.parse_args()
FILE_NAME = args.file
NUM_EPOCHS = args.epoch
test_size = args.test
val_size = args.val
hidden_size = args.hidden
out_size = args.outsize
embedd_dim = args.emb
CAESAR_OFFSET = args.caesar

train(NUM_EPOCHS=NUM_EPOCHS, test_size=test_size, val_size=val_size, FILE_NAME=FILE_NAME, hidden_size=hidden_size,
      out_size=out_size, embedd_dim=embedd_dim, CAESAR_OFFSET=CAESAR_OFFSET)
