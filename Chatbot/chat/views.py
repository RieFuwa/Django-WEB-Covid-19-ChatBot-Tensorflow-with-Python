import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy
import tflearn
import random
import json
import pickle
from tensorflow.python.framework import ops

from django.shortcuts import render, redirect
from chat.models import Room, Message
from django.http import HttpResponse, JsonResponse
# Create your views here.

class chtbt():
    def __init__(self):
            #print("ChatBot, sizinle konuşmak için hazır.(Konuşmayı bitirmek istiyorsan quit yazman yeterli.)")
        self.stemmer = LancasterStemmer()
        ops.reset_default_graph()

        with open("intents.json") as file:  # .JSON dosyasını yüklüyoruz.
            self.data = json.load(file)

        try:
            with open("data.pickle", "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)        # ***********************
        except:
            self.words  = []                                                  # boş listeler oluşturuyorum.
            self.labels = []
            self.docs_x = []
            self.docs_y = []

            for self.intent in self.data["intents"]:                              # .JSON aldığımız verileri çıkartıyoruz.
                for self.pattern in self.intent["patterns"]:
                    self.wrds = nltk.word_tokenize(self.pattern)                  # nltk.word_tokenizer kullanarak bir kelime listesine dönüştüreceğiz .
                    self.words.extend(self.wrds)                                  # Yinelenebilir öğenin tüm öğelerini listenin sonuna ekler.
                    self.docs_x.append(self.wrds)                                 # her bir kalıbı docs_x listemize ve ilişkili etiketini docs_y listesine ekleyeceğiz.
                    self.docs_y.append(self.intent["tag"])

                if self.intent["tag"] not in self.labels:                         # Tüm sohbet baloncukları alıyoruz.
                    self.labels.append(self.intent["tag"])

            self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != "?"]#.lower **
            self.words = sorted(list(set(self.words)))                            # Tüm kelimeleri alır, Sıralar. -.sorted-

            self.labels = sorted(self.labels)
            # Veri ön işlememizin bir sonraki adımında kullanmak için  kelimelerin listesini oluşturacaktır.
            # ------
            self.training = []
            self.output = []
            # Listedeki her pozisyon, kelime dağarcığımızdan bir kelimeyi temsil edecektir.
            # Makine sinir ağı yalnızca sayıları anlar. Sayılar üzerinden çalışarak words bag oluşturuyoruz.
            self.out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(docs_x): #numaralandırma.
                self.bag = []

                self.wrds = [self.stemmer.stem(w.lower()) for w in doc]

                for w in self.words:
                    if w in self.wrds:
                        self.bag.append(1)                                      #  1, kelimemizin var olduğu belirtir.
                    else:
                        self.bag.append(0)                                      # 0 ise o kelimeye sahip olmadığımızı belirtir.
                                                                        # Sadece kelime hazinemizdeki kelimelerin varlığını biliyoruz.
                self.output_row = out_empty[:]
                self.output_row[labels.index(self.docs_y[x])] = 1                    # İlgili etiketde yer alıyor ise çıktı satırı 1.

                self.training.append(self.bag)
                self.output.append(self.output_row)


            self.training = numpy.array(self.training)                               # Tf için numpy dizisi.
            self.output = numpy.array(self.output)                                   #Son olarak eğitim verilerimizi, numpy dizisine dönüştüreceğiz.
        # ---------------------------------------
            with open("data.pickle", "wb") as f: #dosya okuma/yazma.
                pickle.dump((self.words, self.labels, self.training, self.output), f)

        #tensorflow.reset_default_graph()


        # Amacımız, kelimeye bakıp onların ait olduğu bir sınıf vermek olacak. -JSON file.

        # Artık tüm verilerimizi önceden işlediğimize göre, bir model oluşturmaya ve eğitmeye başlamaya hazırız. training ayarları aşagıda*

        self.net = tflearn.input_data(shape=[None, len(self.training[0])])
        self.net = tflearn.fully_connected(self.net, 8) #Burada olan sayılar ile uğraşıp daha iyi bir model oluşturabiliriz. Deneme yanılma.
        self.net = tflearn.fully_connected(self.net, 8) #8 nöronumuz olacak.
        self.net = tflearn.fully_connected(self.net, len(self.output[0]), activation="softmax") #çıktı
        self.net = tflearn.regression(self.net)

        self.model = tflearn.DNN(self.net)
        self.model.load("model.tflearn")

    def bag_of_words(self,s, words):
        self.bag = [0 for _ in range(len(self.words))]

        s_words = nltk.word_tokenize(s)                                 #nltk.word_tokenizer oluşturdugumuz kelime çantamız
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    self.bag[i] = 1 # Kullanıcıdan bir miktar girdi al - Bunu  kelime çantasına dönüştür.

        return numpy.array(self.bag) # numpy array
    # Bag_of_words Sohbet işlevi modeli bir tahmin alır ve JSON dosyasından uygun bir yanıt kapacaktır.

    def chat(self,request):   
        inp = request.POST['message']        

        results = self.model.predict([self.bag_of_words(inp, self.words)])
        results_index = numpy.argmax(results)
        self.tag = self.labels[results_index]                                             #self.self. CBX 'in düşündügü etiket.

        for tg in self.data["intents"]:
            if tg['tag'] == self.tag:
                responses = tg['responses']

        return random.choice(responses) # CBX uygun sohbet boluncuguna gidip random cevap üretir ve yazar.
cb=chtbt()

def home(request):
    return render(request, 'home.html')

def room(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    return render(request, 'room.html', {
        'username': username,
        'room': room,
        'room_details': room_details
    })

def checkview(request):
    room = request.POST['room_name']
    username = request.POST['username']

    if Room.objects.filter(name=room).exists():
        return redirect('/'+room+'/?username='+username)
    else:
        new_room = Room.objects.create(name=room)
        new_room.save()
        return redirect('/'+room+'/?username='+username)

def send(request):
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']

    new_message = Message.objects.create(value=message, user=username, room=room_id)
    new_message.save()
    bot_message = cb.chat(request)
    new_message = Message.objects.create(value=bot_message, user="CDX", room=room_id)
    new_message.save()
    return HttpResponse('Message sent successfully')

def getMessages(request, room):
    room_details = Room.objects.get(name=room)
    messages = Message.objects.filter(room=room_details.id)
    return JsonResponse({"messages":list(messages.values())})       



