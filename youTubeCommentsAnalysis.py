import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from textblob import TextBlob
from multiprocessing import Pool



def sentimentAnalysis():
    pol = []
    sub = []
    comm = pd.read_csv('youTubeVideosUTF.csv',encoding='utf8',\
                       error_bad_lines=False,low_memory=False)
    comm = comm.sample(1000)
    
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
    comm.to_csv('youTubeVideosSentimentAnalysisSample10.csv',sep=',',encoding='utf-8')
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
    
if __name__ == '__main__':
    Pool(sentimentAnalysis())

