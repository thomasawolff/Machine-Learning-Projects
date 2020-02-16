
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

## This class sets up the class data variables that will be passed to subsequent classes through inheritance
class featureEngineering(object):

    def __init__(self,file1,file2):
        ## Constructor creating files to be used in methods
        self.file1 = file1
        self.file2 = file2
        self.biz_df = pd.read_csv(self.file1, engine='python')
        self.df = pd.read_csv(self.file2,delimiter=', ', engine='python')
        ## Compute the log transform of the review count
        self.biz_df['log_review_count'] = np.log10(self.biz_df['review_count'] + 1)
        ## Compute the log transform of the new popularity dataset
        self.df['log_n_tokens_content'] = np.log10(self.df['n_tokens_content'] + 1)



## This class is inheriting class variables set up in the featureEngineering class
class dataTransformNews(featureEngineering):
           
    def linearRegressionNews(self):
        news_orig_model = linear_model.LinearRegression()
        scores_orig = cross_val_score(news_orig_model, self.df[['n_tokens_content']], self.df['shares'], cv=10)

        news_log_model = linear_model.LinearRegression()
        scores_log = cross_val_score(news_log_model, self.df[['log_n_tokens_content']], self.df['shares'], cv=10)

        print('----------News Data----------')
        print("R-squared score without log transform: %0.5f (+/- %0.5f)" % (scores_orig.mean(), scores_orig.std() * 2))
        print("R-squared score with log transform: %0.5f (+/- %0.5f)" % (scores_log.mean(), scores_log.std() * 2))

        ## R-squared score without log transform: -0.00242 (+/- 0.00509)
        ## R-squared score with log transform: -0.00114 (+/- 0.00418)

    def logTransformPlotNews(self):
        ## Visualize the distribution of review counts before and after log transform on the news data
        plt.figure()
        ax = plt.subplot(2,1,1)
        self.df['n_tokens_content'].hist(ax=ax, bins=100)
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Number of Words in Article', fontsize=14)
        ax.set_ylabel('Number of Articles', fontsize=14)

        ax = plt.subplot(2,1,2)
        self.df['log_n_tokens_content'].hist(ax=ax, bins=100)
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Log of Number of Words', fontsize=14)
        ax.set_ylabel('Number of Articles', fontsize=14)
        plt.show()

    def sharesCorrelationNewData(self):
        ## Vizualizing correlations between number of words in article and number of shares from news data
        ## Before log transform
        plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax1.scatter(self.df['n_tokens_content'], self.df['shares'])
        ax1.tick_params(labelsize=14)
        ax1.set_xlabel('Number of Words in Article', fontsize=14)
        ax1.set_ylabel('Number of Shares', fontsize=14)
        ## After log transform
        ax2 = plt.subplot(2,1,2)
        ax2.scatter(self.df['log_n_tokens_content'], self.df['shares'])
        ax2.tick_params(labelsize=14)
        ax2.set_xlabel('Log of the Number of Words in Article', fontsize=14)
        ax2.set_ylabel('Number of Shares', fontsize=14)
        plt.show()



class dataTransformYelp(featureEngineering):

    def linearRegressionYelp(self):
        ## Train linear regression models to predict the average stars rating of a business,
        ## using the review_count feature with and without log transformation
        ## Compare the 10-fold cross validation score of the two models
        m_orig = linear_model.LinearRegression()
        scores_orig = cross_val_score(m_orig, self.biz_df[['review_count']], self.biz_df['stars'], cv=10)
        
        m_log = linear_model.LinearRegression()
        scores_log = cross_val_score(m_log, self.biz_df[['log_review_count']], self.biz_df['stars'], cv=10)

        print('----------Yelp Data----------')
        print("R-squared score without log transform: %0.5f (+/- %0.5f)" % (scores_orig.mean(), scores_orig.std() * 2))
        print("R-squared score with log transform: %0.5f (+/- %0.5f)" % (scores_log.mean(), scores_log.std() * 2))

        ## R-squared score without log transform: -0.00005 (+/- 0.00351)
        ## R-squared score with log transform: 0.00635 (+/- 0.00565)

    def averageStarRatingYelpData(self):
        ## Vizualizing plot of Review Count vs Average Star Rating before and after log transform
        plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax1.scatter(self.biz_df['review_count'], self.biz_df['stars'])
        ax1.tick_params(labelsize=14)
        ax1.set_xlabel('Review Count', fontsize=14)
        ax1.set_ylabel('Average Star Rating', fontsize=14)

        ax2 = plt.subplot(2,1,2)
        ax2.scatter(self.biz_df['log_review_count'], self.biz_df['stars'])
        ax2.tick_params(labelsize=14)
        ax2.set_xlabel('Log of Review Count', fontsize=14)
        ax2.set_ylabel('Average Star Rating', fontsize=14)
        plt.show()

    def logTransformPlotYelp(self):
        ## Visualize the distribution of review counts before and after log transform on the Yelp data
        ax = plt.subplot(2,1,1)
        self.biz_df['review_count'].hist(ax=ax, bins=100)
        ax.tick_params(labelsize=14)
        ax.set_xlabel('review_count', fontsize=14)
        ax.set_ylabel('Occurrence', fontsize=14)

        ax = plt.subplot(2,1,2)
        self.biz_df['log_review_count'].hist(ax=ax, bins=100)
        ax.tick_params(labelsize=14)
        ax.set_xlabel('log10(review_count))', fontsize=14)
        ax.set_ylabel('Occurrence', fontsize=14)
        plt.show()



## This class will perform BoxCox transform on the Yelp data
class dataTransformboxCoxYelp(featureEngineering):

    def paramsSetting(self):
        self.biz_df['review_count'].min()
        
        # Setting input parameter lmbda to 0 gives us the log transform (without constant offset)
        rc_log = stats.boxcox(self.biz_df['review_count'], lmbda=0)
        # By default, the scipy implementation of Box-Cox transform finds the lmbda parameter
        # that will make the output the closest to a normal distribution
        rc_bc, bc_params = stats.boxcox(self.biz_df['review_count'])
        print(round(bc_params,3)) # -0.253

        # setting class variable data to box cox tranform values
        self.biz_df['rc_bc'] = rc_bc
        # setting class variable data to log transform values
        self.biz_df['rc_log'] = rc_log

    def boxCoxHistoPlot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        
        # histogram prior to any transform
        self.biz_df['review_count'].hist(ax=ax1, bins=100)
        ax1.set_yscale('log')
        ax1.tick_params(labelsize=14)
        ax1.set_title('Review Counts Histogram', fontsize=14)
        ax1.set_xlabel('')
        ax1.set_ylabel('Occurrence', fontsize=14)
        
        # histogram after log transform
        self.biz_df['rc_log'].hist(ax=ax2, bins=100)
        ax2.set_yscale('log')
        ax2.tick_params(labelsize=14)
        ax2.set_title('Log Transformed Counts Histogram', fontsize=14)
        ax2.set_xlabel('')
        ax2.set_ylabel('Occurrence', fontsize=14)

        # histogram after optimal Box-Cox transform
        self.biz_df['rc_bc'].hist(ax=ax3, bins=100)
        ax3.set_yscale('log')
        ax3.tick_params(labelsize=14)
        ax3.set_title('Box-Cox Transformed Counts Histogram', fontsize=14)
        ax3.set_xlabel('')
        ax3.set_ylabel('Occurrence', fontsize=14)
        plt.show()

    def probPlotBoxCox(self):
        ## probability plots vs the normal distribution
        fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
        prob1 = stats.probplot(self.biz_df['review_count'], dist=stats.norm, plot=ax1)
        ax1.set_xlabel('')
        ax1.set_title('Probplot against normal distribution')
        prob2 = stats.probplot(self.biz_df['rc_log'], dist=stats.norm, plot=ax2)
        ax2.set_xlabel('')
        ax2.set_title('Probplot after log transform')
        prob3 = stats.probplot(self.biz_df['rc_bc'], dist=stats.norm, plot=ax3)
        ax3.set_xlabel('Theoretical quantiles')
        ax3.set_title('Probplot after Box-Cox transform')
        plt.show()

        

## Data is being pulled from the csv files that I placed into my Github account.
## No need to look for csv files. Run this code from anywhere as long as you have Python 3
url1 = ('https://raw.githubusercontent.com/thomasawolff/verification_text_data/master/YelpReviews10000.csv')
url2 = ('https://raw.githubusercontent.com/thomasawolff/verification_text_data/master/OnlineNewsPopularity.csv')

## Choose a function to call for results. Comment out which one you dont want to use.
def newsCall():
    call = dataTransformNews(url1,url2)
    call.linearRegressionNews()
    call.logTransformPlotNews()
    call.sharesCorrelationNewData()
#newsCall()

def yelpCall():
    call = dataTransformYelp(url1,url2)
    call.linearRegressionYelp()
    call.logTransformPlotYelp()
    call.averageStarRatingYelpData()
#yelpCall()

def boxCoxCall():
    call = dataTransformboxCoxYelp(url1,url2)
    call.paramsSetting()
    call.boxCoxHistoPlot()
    call.probPlotBoxCox()
boxCoxCall()


    
    

    

    
