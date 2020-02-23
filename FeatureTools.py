
import csv
import pandas as pd
import numpy as np
import featuretools as ft
import warnings
warnings.filterwarnings('ignore')

class featureTools(object):

    ## gozar the constructor!!
    def __init__(self,client,loans,payments):
        ## Setting up the data to use throughout the class featureTools
        self.client = client
        self.loans = loans
        self.payments = payments
        ## Reading the csv files from the github site
        self.clientsData = pd.read_csv(self.client, parse_dates = ['joined'])
        self.loansData = pd.read_csv(self.loans, parse_dates = ['loan_start','loan_end'])
        self.paymentsData = pd.read_csv(self.payments, parse_dates = ['payment_date'])
        self.es = ft.EntitySet(id = 'clients')

    def clientsEntity(self):
        ## Create an entity from the client dataframe
        ## This dataframe already has an index and a time index
        self.es = self.es.entity_from_dataframe(entity_id = 'clients', dataframe = self.clientsData,\
                              index = 'client_id', time_index = 'joined')
        ##        print(self.es)
        
        ##        Entityset: clients
        ##          Entities:
        ##            clients [Rows: 25, Columns: 4]
        ##        Relationships:
        ##            No relationships

    def loansEntity(self):
        ## Create an entity from the loans dataframe
        ## This dataframe already has an index and a time index
        self.es = self.es.entity_from_dataframe(entity_id = 'loans', dataframe = self.loansData, 
                              variable_types = {'repaid': ft.variable_types.Categorical},
                              index = 'loan_id', 
                              time_index = 'loan_start')
        ##        print(self.es['loans'])
        
        ##        Entity: loans
        ##          Variables:
        ##            loan_id (dtype: index)
        ##            client_id (dtype: numeric)
        ##            loan_type (dtype: categorical)
        ##            loan_amount (dtype: numeric)
        ##            loan_start (dtype: datetime_time_index)
        ##            loan_end (dtype: datetime)
        ##            rate (dtype: numeric)
        ##            repaid (dtype: categorical)
        ##          Shape:
        ##            (Rows: 443, Columns: 8)
        
    def paymentsEntity(self):
        ## Create an entity from the payments dataframe
        ## This does not yet have a unique index
        self.es = self.es.entity_from_dataframe(entity_id = 'payments', 
                              dataframe = self.paymentsData,
                              variable_types = {'missed': ft.variable_types.Categorical},
                              make_index = True,
                              index = 'payment_id',
                              time_index = 'payment_date')
        ## print(self.es['payments'])
        
        ##
        ##        Entity: payments
        ##          Variables:
        ##            payment_id (dtype: index)
        ##            loan_id (dtype: numeric)
        ##            payment_amount (dtype: numeric)
        ##            payment_date (dtype: datetime_time_index)
        ##            missed (dtype: categorical)
        ##          Shape:
        ##            (Rows: 3456, Columns: 5)

    def dataMerge(self):
        self.clientsEntity()
        self.loansEntity()
        self.paymentsEntity()

        ## Relationship between clients and previous loans
        r_client_previous = ft.Relationship(self.es['clients']['client_id'],
                                            self.es['loans']['client_id'])

        ## Add the relationship to the entity set
        self.es = self.es.add_relationship(r_client_previous)

        ## Relationship between previous loans and previous payments
        r_payments = ft.Relationship(self.es['loans']['loan_id'],
                                      self.es['payments']['loan_id'])

        ## Add the relationship to the entity set
        self.es = self.es.add_relationship(r_payments)

        ## print(self.es)

        ##            Entityset: clients
        ##              Entities:
        ##                clients [Rows: 25, Columns: 4]
        ##                loans [Rows: 443, Columns: 8]
        ##                payments [Rows: 3456, Columns: 5]
        ##              Relationships:
        ##                loans.client_id -> clients.client_id
        ##                payments.loan_id -> loans.loan_id

    def primatives(self):
        self.dataMerge()
        primatives = ft.list_primitives()
        pd.options.display.max_colwidth = 100
        ## print(primatives[primatives['type'] == 'aggregation'].head(10))

        ##         name        ...                            description
        ##0   time_since_last  ...  Calculates the time elapsed since the last datetime (default in seconds).
        ##1             trend  ...                              Calculates the trend of a variable over time.
        ##2  avg_time_between  ...         Computes the average number of seconds between consecutive events.
        ##3      percent_true  ...                                   Determines the percent of `True` values.
        ##4        num_unique  ...           Determines the number of distinct values, ignoring `NaN` values.
        ##5               min  ...                      Calculates the smallest value, ignoring `NaN` values.
        ##6          num_true  ...                                        Counts the number of `True` values.
        ##7               any  ...                               Determines if any value is 'True' in a list.
        ##8            median  ...                      Determines the middlemost number in a list of values.
        ##9              last  ...                                       Determines the last value in a list.

        ## Create new features using specified primitives
        self.features, self.feature_names = ft.dfs(entityset = self.es, target_entity = 'clients', 
                                 agg_primitives = ['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives = ['year','month','diff','divide_by_feature'])

        ## print(pd.DataFrame(self.features['MONTH(joined)'].head()))
        ##  client_id           MONTH(joined)
        ##  42320                  4
        ##  39384                  6
        ##  26945                 11
        ##  41472                 11
        ##  46180                 11

        ## print(pd.DataFrame(self.features['MEAN(payments.payment_amount)'].head()))
        ##  client_id          MEAN(payments.payment_amount)              
        ##  42320                        1021.483333
        ##  39384                        1193.630137
        ##  26945                        1109.473214
        ##  41472                        1129.076190
        ##  46180                        1186.550336

        ## print(self.features.head())

        ##                           income  ...  1 / LAST(payments.payment_amount)
        ##        client_id          ...                                   
        ##        42320      229481  ...                           0.000924
        ##        39384      191204  ...                           0.000489
        ##        26945      214516  ...                           0.000626
        ##        41472      152214  ...                           0.000688
        ##        46180       43851  ...                           0.001435
        
        
    def deepFeatureSynthesis(self):
        self.primatives()
        ## Show a feature with a depth of 2
        ## print(pd.DataFrame(self.features['LAST(loans.MEAN(payments.payment_amount))'].head(10)))
        
        ##                                LAST(loans.MEAN(payments.payment_amount))
        ##      client_id                                           
        ##        42320                                    1192.333333
        ##        39384                                    2311.285714
        ##        26945                                    1598.666667
        ##        41472                                    1427.000000
        ##        46180                                     557.125000
        ##        46109                                    1708.875000
        ##        32885                                    1729.000000
        ##        29841                                    1125.500000
        ##        38537                                    1348.833333
        ##        35214                                    1410.250000

    def autoDeepFeatureSynthesis(self):
        self.dataMerge()
        ## Perform deep feature synthesis without specifying primitives
        self.features, feature_names = ft.dfs(entityset=self.es, target_entity='clients', max_depth = 10)
        print(self.features.iloc[:,4:])
        ##                      STD(loans.rate)  ...            MODE(payments.loans.client_id)
        ##      client_id                     ...                                
        ##        42320             1.984938  ...                           42320
        ##        39384             2.629599  ...                           39384
        ##        26945             1.619717  ...                           26945
        ##        41472             3.198366  ...                           41472
        ##        46180             2.550263  ...                           46180

        ## Using a max depth of 2 produced a dataset with 106 features. I later used a max depth of 10
        ## and a dataset was produced with 146 features.

        

urlClient = ('https://raw.githubusercontent.com/WillKoehrsen/automated-feature-engineering/master/walk_through/data/clients.csv')
urlLoans = ('https://raw.githubusercontent.com/WillKoehrsen/automated-feature-engineering/master/walk_through/data/loans.csv')
urlPayments = ('https://raw.githubusercontent.com/WillKoehrsen/automated-feature-engineering/master/walk_through/data/payments.csv')


feat1 = featureTools(urlClient,urlLoans,urlPayments)
feat1.autoDeepFeatureSynthesis()
