import pandas as pd
import re
import json
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

stopwords = []
stopwords = list(open("./data/stopwords-nlp-vi.txt", encoding="UTF-8-sig", mode="r"))
for i in range(len(stopwords)):
    stopwords[i] = re.sub("\n", "", stopwords[i])

#Đọc các từ viết tắt từ file sang dict

f = pd.read_csv("./data/acronym_vi.txt", sep="\t").to_numpy()
acronyms = {line[0]:line[1] for line in f}

#Chuẩn hóa unicode sang chuẩn unicode dựng sẵn
# def covert_unicode(txt):
#     return re.sub(
#         r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
#         lambda x: dicchar[x.group()], txt)

#Xử lý từ viết tắt
def replace_acronyms(txt):
    pat = re.compile(r"\b(%s)\b" % "|".join(acronyms))
    txt = pat.sub(lambda m: acronyms.get(m.group()), txt)
    return txt

#Xử lý các từ lặp
def remove_loop_char(txt):
    txt = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), str(txt), flags=re.IGNORECASE)
    txt = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',txt)
    return txt

#Xử lý các từ dừng
def remove_stopwords(txt):
    split_words = txt.split()
    final_txt = []
    for ch in split_words:
        if ch not in stopwords:
            final_txt.append(ch)
    return " ".join(final_txt)

#Xử lý dấu câu
def remove_punctuations(txt):
    punctuations = '@#!?+&*[]-%:/();$=><|{}^_' + "'`"
    for p in punctuations:
        txt = txt.replace(p, f' {p} ')
    return txt

#Tiền xử lý 
def preProcessing(txt):
    txt = txt.lower()
    #Xóa các kí tự xuống dòng
    txt = " ".join(re.sub("\n", " ", txt).split())
    #Xử lý dấu câu
    txt = remove_punctuations(txt)
    #Thay thế các từ viết tắt
    txt = replace_acronyms(txt)
    #Xóa các kí tự lặp
    # txt = remove_loop_char(txt)
    #Tách từ
    txt = ViTokenizer.tokenize(txt)
    #Xóa từ dừng
    # txt = remove_stopwords(txt)
    #Chuẩn hóa sang 1 kiểu unicode
    # txt = covert_unicode(txt)
    txt = txt.lower()
    # Xóa bớt các khoảng trắng thừa
    return " ".join(txt.split())
