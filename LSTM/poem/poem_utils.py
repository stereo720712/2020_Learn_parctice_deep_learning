import numpy as np
import collections

END_CHAR = '\n'

UNKNOWN_CHAR = ' '
MIN_LENGTH = 10

#sentence length
unit_sentence = 6
max_words = 3000

#load from txt file
def load(poetry_file):
    def handle(line):
        return line + END_CHAR

    poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
               open(poetry_file,encoding='utf-8')]

    collect = []

    for poetry in poetrys:
        if len(poetry) <= 5:
            continue
        if poetry[5] == "，" :
            collect.append(handle(poetry))

    print(len(collect))
    poetrys = collect

    # 所有字
    words = []
    for poetry in poetrys:
        words += [word for word in poetry]
    # get all words ,show times sorted from max to min
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words,_ = zip(*count_pairs)

    #取出頻率高的字組成字典 頻率不高的用" "代替
    words_size = min(max_words,len(words))
    words = words[:words_size] + (UNKNOWN_CHAR,)
    # calculate total length
    words_size = len(words)

    #word map to id , use one-hot
    char2id_dict = {w:i for i,w in enumerate(words)}
    id2char_dict = {i:w for i, w in enumerate(words)}

    unknow_char = char2id_dict[UNKNOWN_CHAR]
    char2id = lambda char:char2id_dict.get(char,unknow_char)
    poetrys = sorted(poetrys,key=lambda line:len(line))
    #訓練集中每一首詩 都找到 每個字對應的id
    poetrys_vector = [list(map(char2id,poetry)) for poerty in poetrys]
    return np.array(poetrys_vector),char2id_dict,id2char_dict

def get_6t01(x_data,char2id_dict):
    # 讀取每首詩中 所有6個字學到他後一個字的對應 ？？
    inputs = []
    targets = []
    for index in range(len(x_data)):
        x = x_data[index:(index+unit_sentence)]
        y = x_data[index + unit_sentence]
        if (END_CHAR in x) or y == char2id_dict[END_CHAR]:
            return np.array(inputs),np.array(targets)
        else:
            inputs.append(x)
            targets.append(y)

    return np.array(inputs),np.array(targets)

# https://blog.csdn.net/learning_tortosie/article/details/85243310
# encoding  inputs
def get_batch(batch_size,x_data,char2id_dict,id2char_dict):
    n = len(x_data)
    batch_i = 0
    words_size = len(char2id_dict)
    while(True):
        one_hot_x_data = []
        one_hot_y_data = []
        #對 batch_size 首詩做 6to1 操作
        for i in range(batch_size):
            batch_i = (batch_i+1)%n
            inputs,targets = get_6t01(x_data[batch_i],char2id_dict)
            for j in range(len(inputs)):
                one_hot_x_data.append(inputs[j])
                one_hot_y_data.append(targets[j])

        batch_size_after = len(one_hot_x_data)

        #將6to1的結果 變化為6to1的形式
        input_data = np.zeros((batch_size_after,unit_sentence,words_size))
        target_data = np.zeros((batch_size_after,words_size))

        for i , (input_text,target_text) in enumerate(zip(one_hot_x_data,one_hot_y_data)):
            #將輸入得每一個step對應的字的id對應的那個位置設置為1
            for t,index in enumerate(input_text):
                input_data[i,t,index] = 1
                #將輸出的字的id對應的那個位置設置為一
                target_data[i,target_text] = 1.

        yield input_data,target_data

def predict_from_nothing(epoch,x_data,char2id_dict,id2char_dict,model):
    #progress training , print learning status
    print("\n#-----------------------Epoch {}--------------------#".format(epoch))
    words_size = len(id2char_dict)

    #隨機抓取一首詩的開頭6個字符,進行後面的預測
    index = np.random.randint(0,len(x_data))
    sentence = x_data[index][:unit_sentence]

    def _pred(text):
        temp = text[-unit_sentence:]
        x_pred = np.zeros((1,unit_sentence,words_size))
        for t , index in enumerate(temp):
            x_pred[0,t,index] = 1.

        preds = model.predict(x_pred)[0]
        choice_id = np.random.choice(range(len(preds)),1,p=preds)
        if id2char_dict[choice_id[0]] == ' ':
            while id2char_dict[choice_id[0]] in ['，','。',' ']:
                    choice_id = np.random.randint(0,len(char2id_dict),1)
        return  choice_id

    #一個字一個字往後預測
    for i in range(24 - unit_sentence):
        pred = _pred(sentence)
        sentence = np.append(sentence,pred)

    output = ""

    for  i in range(len(sentence)):
        output = output + id2char_dict[sentence[i]]

    print(output)


def predict_from_head(name,x_data,char2id_dict,id2char_dict,model):

    #根據給定的字,生成藏頭詩

    if len(name) < 4:
        for i in range(4-len(name)):
            index = np.random.randint(0,len(char2id_dict))
            while id2char_dict[index] in ['，','。',' ']:
                index = np.random.randint(0,len(char2id_dict))

            name += id2char_dict[index]

    origin_name = name
    name = list(name)

    for i in range(len(name)):
        if name[i] not in char2id_dict:
            index = np.random.randint(0,len(char2id_dict))
            while id2char_dict[index] in ['，','。',' ']:
                index = np.random.randint(0,len(char2id_dict))
            name[i] = id2char_dict[index]

    name = ''.join(name)
    words_size = len(char2id_dict)
    index = np.random.randint(0,len(x_data))

    # 選取隨機一首詩的最後max_len字符+給出的首個為字作為初始輸入
    sentence = np.append(x_data[index][-unit_sentence:-1],char2id_dict[name[0]])

    #?!repeat , 跟上面一樣
    def _pred(text):
        temp = text[-unit_sentence:]
        x_pred = np.zeros((1,unit_sentence,words_size))
        for t , index in enumerate(temp):
            x_pred[0,t,index] = 1

        preds = model.predict(x_pred)[0]

        choice_id = np.random.choice(range(len(preds)),1,p=preds)
        if id2char_dict[choice_id[0]] == " ":
            while id2char_dict[choice_id[0]] in ['，', '。', ' ']:
                choice_id = np.random.randint(0, len(char2id_dict), 1)
        return choice_id

    # 首先 預測出包含藏頭詩第一個字的詩的前六個字
    for i in range(5):
        pred = _pred(sentence)
        sentence = np.append(sentence,pred)

     #然後利用這六個字繼續往下預測
    sentence = sentence[-unit_sentence:]
    for i in range(3):
        sentence = np.append(sentence,char2id_dict[name[i+1]])
        for i in range(5):
            pred = _pred(sentence)
            sentence = np.append(sentence,pred)


    #保證藏頭正確
    output = []
    for i in range(len(sentence)):
        output.append(id2char_dict[sentence[i]])
    for i in range(4):
        output[i*6] = origin_name[i] # restore sentence first word

    output = ''.join(output)
    print(output)





