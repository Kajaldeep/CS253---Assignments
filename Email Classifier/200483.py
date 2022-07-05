import csv
import sys
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import svm
csv.field_size_limit(sys.maxsize)

email = []
label = []
stem_words = []
nonstem_words = []
spam_bow = []
nonspam_bow = []
vocab = []
train_data = []
test_data = []
bow = []

def read_email(lst):
  return lst.split()

def stemming(words):
  ps = PorterStemmer()
  stemwords = []
  for w in words:
    stemwords.append(ps.stem(w))
  return stemwords

def remove_stop_words(stem_words):
  stop_words = set(stopwords.words('english'))
  stem_no_stop_words = []
  for w in stem_words:
    if w not in stop_words:
        stem_no_stop_words.append(w)
  return stem_no_stop_words

def build_vocabulary(wrd):
  for i in wrd:
    if i not in vocab:
      vocab.append(i)
  return vocab

def get_bow():
  for j in vocab:
    # if(j in nonstem_words):
      nonspam_bow.append(nonstem_words.count(j))
    # else:
    #   nonspam_bow.append(0)
    # if(j in stem_words):
      spam_bow.append(stem_words.count(j))
    # else:
    #   spam_bow.append(0)
  return nonspam_bow, spam_bow

# split dataset
def split():
  index = np.arange(3000)
  idx = (np.random.choice(index, size=600, replace = False)).tolist()
  x_train = []
  x_test = []
  y_train = []
  y_test = []
  for i in range(3000):
    if i in idx:
      x_test.append(email[i])  
      y_test.append(label[i])
    else:
      x_train.append(email[i])
      y_train.append(label[i])
  # x_train, x_test, y_train, y_test = train_test_split(email, label, test_size=0.20, random_state=50)
  train_data.append(x_train)
  train_data.append(y_train)
  test_data.append(x_test)
  test_data.append(y_test)
  return train_data, test_data

def read_data():
  i=0
  for rf in (email):
    wrd = remove_stop_words(stemming(read_email(str(rf))))
    if(label[i] == "0"):
      nonstem_words.extend(wrd)
    else:
      stem_words.extend(wrd)
    vocab = build_vocabulary(wrd)
    i = i + 1

  nonspam_bow, spam_bow = get_bow()
  for i in range(0, len(vocab)):
    bow.append(spam_bow[i] + nonspam_bow[i])

def data_vis():
    plt.plot(vocab,bow)
    plt.xlabel('vocab')
    plt.ylabel('frequency')
    plt.title('vocab-frequency')
    plt.show()

  # nonspam_words
    plt.plot(vocab,nonspam_bow)
    plt.xlabel('nonspam words')
    plt.ylabel('frequency')
    plt.title('nonspam_words-frequency')
    plt.show()

  # # spam mails 
    plt.plot(vocab,spam_bow)
    plt.xlabel('spam words')
    plt.ylabel('frequency')
    plt.title('spam_words-frequency')
    plt.show()
    return

def sel_data(data):
  freq = np.zeros((len(data[0]), len(vocab)))
  for i in range(len(data[0])):
    for j in range(len(vocab)):
      freq[i][j] = int((data[0])[i].count(vocab[j]))
  return freq


def svm_classifier():
  cls = svm.SVC(kernel="linear")
  cls.fit(freq_train, train_data[1])
  res = cls.predict(freq_test)
  print("Prediction of svm : ", res.tolist())
  print("Accuracy of svm : " , compute_accuracy(res.tolist(), test_data[1]))
  print("AUC score svm : ",compute_auc(res.tolist(), test_data[1]))


def knn_classifier():
      neigh = KNeighborsClassifier(n_neighbors=3)
      neigh.fit(freq_train, train_data[1])
      res = neigh.predict(freq_test)
      print("Prediction of knn : ", res.tolist())
      print("Accuracy of knn : " ,compute_accuracy(res.tolist(), test_data[1]))
      print("AUC score knn : " , compute_auc(res.tolist(), test_data[1]))

# compute accuracy 
def compute_accuracy(true_labels, predicted_labels):
  acc_score = 0
  for i in range(len(true_labels)):
        if(true_labels[i] == predicted_labels[i]):
              acc_score += 1
  acc_score = (acc_score/len(true_labels))
  return acc_score

# compute AUC score 
def compute_auc(true_labels, predicted_labels):
      return roc_auc_score(true_labels, predicted_labels, average=None)


if _name_ == "_main_":
    # reading CSV file
    file = open('spam_or_not_spam.csv', encoding='utf-8')
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    for i in range(len(rows)):
        email.append((rows[i])[0])
        label.append((rows[i])[1])
    
    read_data()
    data_vis()


    train_data, test_data = split()

    freq_train = sel_data(train_data)
    freq_test = sel_data(test_data)
  
    svm_classifier()
    knn_classifier()