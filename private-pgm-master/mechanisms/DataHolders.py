import numpy as np
import pandas as pd


class HDataHolder:
    def __init__(self, name, dataset, partition,n):
        assert n < 1
        self.name = name
        self.dataset = dataset
        self.partition = partition
        self.n = n
        # private info
        self.data = None
        self.workload_answers = None
        self.total_samples = 0
        self.weights = None

    def load_data(self):
        self.data = pd.read_csv(self.dataset)
        '''
        if self.name == "Alice":
            #if self.partition == "horizontal":
                n = int(df.shape[0] * self.n)
                self.data = df.iloc[0:n,:]
        elif self.name == "Bob":
            #if self.partition == "horizontal":
                n = int(df.shape[0] * self.n)
                self.data = df.iloc[n:df.shape[0],:]
        '''

    def number_of_samples(self):
        self.total_samples = self.data.shape[0]

    def compute_answers(self,workload,domain,max_domain_size,keep_pad=True,flatten=True):
        my_ans = []
        for cl in workload:
            shape = domain.project(cl).shape
            bins = [range(n+1) for n in shape]
            ans = np.histogramdd(self.data[list(cl)].values, bins, weights=self.weights)[0]
            data_vector = ans.flatten() if flatten else ans
            if keep_pad:
                padded_data_vector = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
                my_ans.append(padded_data_vector)
            else:
                my_ans.append(data_vector)
        self.workload_answers = my_ans


class VDataHolder:
    def __init__(self, name, dataset, partition,n):
        assert n < 1
        self.name = name
        self.dataset = dataset
        self.partition = partition
        self.n = n
        # private info
        self.data = None
        self.workload_answers = None
        self.total_samples = 0
        self.total_columns = 0
        self.weights = None
        self.workload_ids = []
        self.column_ids_for_workload_ids = []


    def load_data(self,workload):
        self.data = pd.read_csv(self.dataset)

        '''
        self.total_columns = df.shape[1]
        if self.name == "Alice":
                n = int(df.shape[1] * self.n)
                self.data = df.iloc[:,0:n]

        elif self.name == "Bob":
                n = int(df.shape[1] * self.n)
                self.data = df.iloc[:,n:df.shape[1]]
        '''
        self.workload_ids = self.get_workload_ids(workload)
        # self.column_ids_for_workload_ids = self.get_column_ids_workload_ids(workload)


    def number_of_samples(self):
        self.total_samples = self.data.shape[0]


    def compute_answers(self,workload,domain,max_domain_size,flatten=True, padded=True):
        my_ans = np.zeros((len(workload), max_domain_size))
        #my_ans = np.zeros(len(workload))
        for i,cl in enumerate(workload):
            if i in self.workload_ids:
                shape = domain.project(cl).shape
                bins = [range(n+1) for n in shape]
                ans = np.histogramdd(self.data[list(cl)].values, bins, weights=self.weights)[0]
                data_vector = ans.flatten() if flatten else ans
                if padded:
                    data_vector = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
                my_ans[i]= data_vector
        self.workload_answers = my_ans

    def get_noisy_measurement(self, ax, ax_index, scale, domain):
        size = domain.project(ax).size()
        y = self.workload_answers[ax_index,:size] + np.random.normal(loc=0, scale=scale, size=size)
        return y

    def get_workload_ids(self, workload):
        columns = self.data.columns
        workload_ids = []
        for i,cl in enumerate(workload):
            if all(elem in columns for elem in cl):
                workload_ids.append(i)
        return workload_ids

    def get_column_ids_workload_ids(self, workload):
        columns = self.data.columns
        N = self.total_columns - self.data.shape[1]
        column_ids = []
        for i,cl in enumerate(workload):
            if self.name == 'Alice':
                for elem in cl:
                    c = columns.get_index(elem)
                column_ids[i] = [columns.get_index(elem) for elem in cl]
            if self.name == 'Bob':
                column_ids[i] = [N-columns.get_index(elem) for elem in cl]
        return column_ids



