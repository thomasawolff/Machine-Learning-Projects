import re
import nltk
import math
import nltk.corpus
import operator
import pandas as pd
import seaborn as sns
from nltk import ne_chunk
from textblob import TextBlob
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import norm
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer



class textAnalytics(object):

    def __init__(self,file1):
        self.limit = 100
        self.stringsList = []
        self.file1 = file1
        self.review_df = pd.read_csv(self.file1,low_memory=False)
        #self.review_df = self.review_df['commentText']
        self.token_pattern = '(?u)\\b\\w+\\b'
        self.field = 'commentText'
        #print(list(self.review_df))
        self.review_df = self.review_df[['videoID','categoryID','views','likes','dislikes',\
                                         'commentCount','commentText','commentLikes','replies']]
        self.stopWords = stopwords.words('english')
        #print(self.stopWords)
        
    def bowConverter(self):
        bow_converter = CountVectorizer(token_pattern=self.token_pattern)
        x = bow_converter.fit_transform(self.review_df[self.field])
        self.words = bow_converter.get_feature_names()
        #print(len(words)) ## 29221
        
    def biGramConverter(self):
        bigram_converter = CountVectorizer(ngram_range=(2,2), token_pattern=self.token_pattern)
        x2 = bigram_converter.fit_transform(self.review_df[self.field])
        self.bigrams = bigram_converter.get_feature_names()
        #print(len(bigrams)) ## 368937
        #print(bigrams[-10:])
        ##        ['zuzu was', 'zuzus room', 'zweigel wine'
        ##       , 'zwiebel kräuter', 'zy world', 'zzed in'
        ##       , 'éclairs napoleons', 'école lenôtre', 'ém all', 'òc châm']

    def triGramConverter(self):
        trigram_converter = CountVectorizer(ngram_range=(3,3), token_pattern=self.token_pattern)
        x3 = trigram_converter.fit_transform(self.review_df[self.field])
        self.trigrams = trigram_converter.get_feature_names()
        print(len(self.trigrams)) # 881609
        #print(self.trigrams[:10])
        ##        ['0 0 eye', '0 20 less', '0 39 oz', '0 39 pizza', '0 5 i'
        ##         , '0 50 to', '0 6 can', '0 75 oysters', '0 75 that', '0 75 to']

    def gramPlotter(self):
        self.bowConverter()
        self.biGramConverter()
        self.triGramConverter()
        
        sns.set_style("darkgrid")
        counts = [len(self.words), len(self.bigrams), len(self.trigrams)]
        plt.plot(counts, color='cornflowerblue')
        plt.plot(counts, 'bo')
        plt.margins(0.1)
        plt.xticks(range(3), ['unigram', 'bigram', 'trigram'])
        plt.tick_params(labelsize=14)
        plt.title('Number of ngrams in the first 10,000 reviews of the dataset', {'fontsize':16})
        plt.show()

    def wordLem(self):
        self.bowConverter()
        for line in self.words:
            print(line+":"+lemmatizer.lemmatize(line))

    def wordCount(self):
        for line in self.review_df[self.field]:
            wordsTokens = word_tokenize(line)
            self.stringsList.append(Counter(wordsTokens))
        ##  Counter({'.': 11, 'the': 9, 'and': 8, 'was': 8, 'It': 5, 'I': 5, 'it': 4, 'their': 4

    def stringCleaning(self):
        self.wordCount()
        lengthList = []
        punctuationList = ['-?','!',',',':',';','()',"''",'.',"``",'|','^','..','...','--','=']
        for i in range(0,self.limit):
            for words in self.stringsList[i]:
                if len(words)>0:
                    lengthList.append(words)
        post_punctuation = [word for word in lengthList if word not in punctuationList]
        noStopWords = [word for word in post_punctuation if word not in self.stopWords]
        self.postPunctCount = Counter(noStopWords)
        #print(self.postPunctCount)
        ##        Counter({'I': 9, "n't": 6, 'The': 5, 'go': 5, 'good': 5, "'s": 5,
        ##                 'My': 4, 'It': 4, 'place': 4, 'menu': 4, ')': 4, 'outside': 3,
        ##                 'food': 3, 'like': 3, "'ve": 3, 'amazing': 3, 'delicious': 3,
        ##                 'came': 3, 'wait': 3, 'back': 3, 'They': 3, 'evening': 3, 'try': 3,
        ##                 'one': 3, '(': 3, 'awesome': 3,'much': 3, 'took': 2, 'made': 2,
        ##                 'sitting': 2, 'Our': 2, 'arrived': 2, 'quickly': 2, 'looked': 2, ....

    def tagsMaker(self):
        # If you want to run this code, install Ghostscript first
        self.stringCleaning()
        tags = nltk.pos_tag(self.postPunctCount)
        grams = ne_chunk(tags)
        grammers = r"NP: {<DT>?<JJ>*<NN>}"
        chunk_parser = nltk.RegexpParser(grammers)
        chunk_result = chunk_parser.parse(grams)
        print(chunk_result)
        ##        (ORGANIZATION General/NNP Manager/NNP Scott/NNP Petello/NNP)
        ##          (NP egg/NN)
        ##          Not/RB
        ##          (NP detail/JJ assure/NN)
        ##          albeit/IN
        ##          (NP rare/JJ speak/JJ treat/NN)
        ##          (NP guy/NN)
        ##          (NP respect/NN)
        ##          (NP state/NN)
        ##          'd/MD
        ##          surprised/VBN
        ##          walk/VB
        ##          totally/RB
        ##          satisfied/JJ
        ##          Like/IN
        ##          always/RB
        ##          say/VBP
        ##          (PERSON Mistakes/NNP)

    def sentimentAnalysis(self):
        pol = []
        sub = []
        comm = self.review_df 
        comm = comm.sample(100)
        for i in comm.commentText.values:
            try:
                analysis = TextBlob(i)
                pol.append(round(analysis.sentiment.polarity,2))
            except:
                pol.append(0)

        for i in comm.commentText.values:
            try:
                analysis = TextBlob(i)
                sub.append(round(analysis.sentiment.subjectivity,2))
            except:
                sub.append(0)

        comm['polarity']=pol
        comm['subjectivity']=sub
        #comm.to_csv('youTubeVideosSentimentAnalysisSample10.csv',sep=',',encoding='utf-8')
        print(comm)

        plt.grid(axis='y', alpha=0.50)
        plt.title('Histogram of comment sentiment')
        plt.xlabel('Sentiment Scores')
        plt.ylabel('Frequency')
        sns.distplot(comm['polarity'],hist=False,fit=norm,kde=True,norm_hist=True)
        plt.show()

        plt.grid(axis='y', alpha=0.50)
        plt.title('Histogram of comment sentiment')
        plt.xlabel('Sentiment Scores')
        plt.ylabel('Frequency')
        sns.distplot(comm['subjectivity'],hist=False,fit=norm,kde=True,norm_hist=True)
        plt.show()
        ##                    videoID       categoryID  views  ...    replies  polarity   subjectivity
        ##          251449  LLGENw4C1jk          17   1002386  ...      0.0      0.50          0.50
        ##          39834   3VVnY86ulA8          22    802134  ...      0.0      0.00          0.10
        ##          203460  iA86imHKCMw          17   3005399  ...      0.0     -0.08          0.69
        ##          345225  RRkdV_xmYOI          23    367544  ...      0.0      0.13          0.76
        ##          402953  vQ3XgMKAgxc          10  51204658  ...      0.0      0.25          0.50


url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')

go = textAnalytics(url)
#go.bowConverter()
#go.wordCount()
#go.triGramConverter()
##go.gramPlotter()
#go.wordLem()
##go.wordCount()
#go.stringCleaning()
##go.tagsMaker()
##go.computeIDF()
if __name__ == '__main__':
    Pool(go.sentimentAnalysis())
#go.sentimentAnalysis()

