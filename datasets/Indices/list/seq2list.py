import random

def shuffle_train_list(seq_len=5):
    path = 'train_list.csv'
    newFile = 'train_shuffle_list.csv'
    ff = open(newFile ,'w')
    seq_list = []
    idx = 0
    with open(path) as file_obj:
        items = []
        for line in file_obj:
            items.append(line)
            idx += 1
            if idx % seq_len ==0:
                seq_list.append(items)
                items = []

    random.shuffle(seq_list)
    for items in seq_list:
        if len(items)==seq_len:
            for line in items:
                ff.write(line)


def list_to_seq(seq_len=5):
    path = 'train_list.csv'
    newFile = 'train_seq_list.txt'
    ff = open(newFile ,'w')

    new_line = ''
    idx = 0
    with open(path) as file_obj:
        for line in file_obj:
            idx += 1
            new_line = new_line + line.replace('\n' ,';')
            if idx % seq_len ==0:
                new_line = new_line.rstrip(';')
                new_line = new_line +'\n'
                ff.write(new_line)
                new_line = ''

def seq_to_list():
    path = 'train_seq_list.txt'
    newFile = 'train_list.csv'
    ff = open(newFile,'w')

    with open(path) as file_obj:
        for line in file_obj:
            line = line.replace('\n','')
            contents = line.split(';')
            for item in contents:
                item = item+'\n'
                ff.write(item)


if __name__ == '__main__':
    shuffle_train_list(10)