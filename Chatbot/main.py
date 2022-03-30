import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import json
import pickle
from tensorflow.python.framework import ops
ops.reset_default_graph()

with open("intents.json") as file:  # .JSON dosyasını yüklüyoruz.
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)        # ***********************
except:
    words = []                                                  # boş listeler oluşturuyorum.
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:                              # .JSON aldığımız verileri çıkartıyoruz.
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)                  # nltk.word_tokenizer kullanarak bir kelime listesine dönüştüreceğiz .
            words.extend(wrds)                                  # Yinelenebilir öğenin tüm öğelerini listenin sonuna ekler.
            docs_x.append(wrds)                                 # her bir kalıbı docs_x listemize ve ilişkili etiketini docs_y listesine ekleyeceğiz.
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:                         # Tüm sohbet baloncukları alıyoruz.
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]#.lower **
    words = sorted(list(set(words)))                            # Tüm kelimeleri alır, Sıralar. -.sorted-

    labels = sorted(labels)
    # Veri ön işlememizin bir sonraki adımında kullanmak için  kelimelerin listesini oluşturacaktır.
    # ------
    training = []
    output = []
    # Listedeki her pozisyon, kelime dağarcığımızdan bir kelimeyi temsil edecektir.
    # Makine sinir ağı yalnızca sayıları anlar. Sayılar üzerinden çalışarak words bag oluşturuyoruz.
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x): #numaralandırma.
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)                                      #  1, kelimemizin var olduğu belirtir.
            else:
                bag.append(0)                                      # 0 ise o kelimeye sahip olmadığımızı belirtir.
                                                                   # Sadece kelime hazinemizdeki kelimelerin varlığını biliyoruz.
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1                    # İlgili etiketde yer alıyor ise çıktı satırı 1.

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)                               # Tf için numpy dizisi.
    output = numpy.array(output)                                   #Son olarak eğitim verilerimizi, numpy dizisine dönüştüreceğiz.
 # ---------------------------------------
    with open("data.pickle", "wb") as f: #dosya okuma/yazma.
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()


# Amacımız, kelimeye bakıp onların ait olduğu bir sınıf vermek olacak. -JSON file.

# Artık tüm verilerimizi önceden işlediğimize göre, bir model oluşturmaya ve eğitmeye başlamaya hazırız. training ayarları aşagıda*

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) #Burada olan sayılar ile uğraşıp daha iyi bir model oluşturabiliriz. Deneme yanılma.
net = tflearn.fully_connected(net, 8) #8 nöronumuz olacak.
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #çıktı
net = tflearn.regression(net)

model = tflearn.DNN(net)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)                                 #nltk.word_tokenizer oluşturdugumuz kelime çantamız
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1 # Kullanıcıdan bir miktar girdi al - Bunu  kelime çantasına dönüştür.

    return numpy.array(bag) # numpy array
# Bag_of_words Sohbet işlevi modeli bir tahmin alır ve JSON dosyasından uygun bir yanıt kapacaktır.


def chat():
    print("ChatBot, sizinle konuşmak için hazır.(Konuşmayı bitirmek istiyorsan quit yazman yeterli.)")
    while True:
        inp = input("Sen: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]                                             # CBX 'in düşündügü etiket.

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses)) # CBX uygun sohbet boluncuguna gidip random cevap üretir ve yazar.

secim=int(input("Secim yapın Chatbot için 1, eğitim için 2: "))
if(secim==1):
    model.load("model.tflearn")
    chat()
if(secim==2):                                                                   # Artık tüm verilerimizi önceden işlediğimize göre, bir model oluşturmaya ve eğitmeye başlamaya hazırız.
    model.fit(training, output, n_epoch=20000, batch_size=8, show_metric=True)   # Belirlediğimiz periyot sayısı, modelin kaç kere eğitim yapacagı sayısıdır.
    model.save("model.tflearn")
    # Modeli eğitmeyi bitirdiğimizde , diğer komut dosyalarında kullanmak için model.tflearn dosyasına kaydediyoruz.


