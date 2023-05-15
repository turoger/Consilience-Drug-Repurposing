import os, torch
import pandas as pd
import numpy as np


class ProcessOutput(object):
    """
    Scores output processing functions for KG Embedding Methods

    Inputs:
    -----------
    - data_dir              * directory to the data (train/test/valid.txt)
    - scores_outfile        * path to the *_scores.tsv
    - mode                  * head-batch or tail-batch predictions

    Functions:
    ------------
    - format_raw_scores_to_df(self, top): converts raw tsv to formatted tsv

    """
    def __init__(self, data_dir, scores_outfile, mode):
        self.data_dir = data_dir
        self.scores_outfile = scores_outfile
        self.mode = mode
        self.df = pd.read_csv(self.scores_outfile, names = ['h','r','t','preds','batch'], sep = '\t')
        if self.mode not in ['head-batch','tail-batch']:
            raise Exception(f'{mode} not a valid option for argument "mode"')

    def format_raw_scores_to_df(self, df = None ,top: int = None):
        """
        Takes 'test_scores.tsv' and returns entity/relation labeled dataframe
        Inputs
        ------------
        - self.df           * scores file
        - top               * keeps top 'n' predictions
        """
        if df == None:
            df = self.df
        df["preds"] = df["preds"].apply(
            lambda x: torch.argsort(
                torch.tensor([float(i) for i in x.strip("[]").split(",")]),
                descending=True,
            ).tolist()[0:top]
        )
        self.df = df
        return df
    
    def get_true_targets(self, true_rel = 'indication'):
        """
        gets a list of targets for a given doublet {(h,r) | (r,t)} of a triple.

        Inputs
        -----------
        - self.data_dir     * directory of data folder
        - self.mode         * {'tail-batch','head-batch'}
                            'tail-batch' mode grabs a list of tails of a triple as the true values.
                            'head-batch' grabs the list of head entities of a given tail triple
        Outputs
        -----------
        a dataframe to lookup list of head/tail given the mode
        """
        data_dir = self.data_dir

        train = pd.read_csv(
            os.path.join(data_dir, "train.txt"), sep="\t", names=["h", "r", "t"]
        )
        test = pd.read_csv(
            os.path.join(data_dir, "test.txt"), sep="\t", names=["h", "r", "t"]
        )
        valid = pd.read_csv(
            os.path.join(data_dir, "valid.txt"), sep="\t", names=["h", "r", "t"]
        )

        # three dataframes combined vertically. headers are head, relation and tail (triples)
        all_true_triples = pd.concat([train, test, valid])

        if self.mode == "tail-batch":
            # groups head and relation columns together and collapses all items in the df matching
            # the head and relation into the 'tail' column
            df = (
                all_true_triples.groupby(["h", "r"])
                .agg({"t": lambda x: list(x)})
                .reset_index()
            )
        elif self.mode == "head-batch":
            df = (
                all_true_triples.groupby(["r", "t"])
                .agg({"h": lambda x: list(x)})
                .reset_index()
            )
            
        df = df.query('r==@true_rel')
        
        return df
    
    def translate_embeddings(self, df = None, direction: str = "to"):
        """
        Translates dataframe to and from embeddings, depending on the mode
        Function: format_raw_scores_to_df() must be run first
        Inputs
        -----------
        - self.data_dir     * directory of data folder
        - self.df           * dataframe to translate embeddings
        - mode              * {'to','from'}
                              'to'(embedding): Alphabet -> Number
                              'from'(embedding): Number -> Alphabet
        Outputs
        -----------
        a translated dataframe based on the mode. In-place.
        """
        if df == None:
            df = self.df
        data_dir = self.data_dir
        
        if type(df['preds'][0])== str:
            raise Exception('DataFrame needs to have "format_raw_scores_to_df()" run first')
            
        # read in files
        entity_df = pd.read_csv(os.path.join(data_dir, "entities.dict"), sep="\t", names=["emb", "identifier"])
        relation_df = pd.read_csv(os.path.join(data_dir, "relations.dict"), sep="\t", names=["emb", "identifier"])

        # process the options
        if direction == "to":
            entity_dict = dict(zip(entity_df.identifier, entity_df.emb))
            relation_dict = dict(zip(relation_df.identifier, relation_df.emb))
        elif direction == "from":
            entity_dict = dict(zip(entity_df.emb, entity_df.identifier))
            relation_dict = dict(zip(relation_df.emb, relation_df.identifier))
        else:
            raise Exception(f'{mode} is not a valid option for argument "mode"')
        
        for col in df.columns:
            if col == "h" or col == "t":
                df[col] = df[col].apply(lambda x: entity_dict[x])
            elif col == "r":
                df[col] = df[col].apply(lambda x: relation_dict[x])
            elif col == "preds":
                df[col] = df[col].apply(lambda x: [entity_dict[i] for i in x])
            else:
                pass
        self.df = df
        return df
    
    def filter_predictions(self, top: int = 50):
        """
        Get filtered predictions that don't exist as triples in Train/Test/Valid

        Inputs
        -----------
        - self.df           * dataframe to translate embeddings
        - self.mode         * {'head-batch','tail-batch'}
                              'head-batch': given (r,t) predict h
                              'tail-batch': given (h,r) predict t
        - top               * get the top 'n' results for each prediction
        """
        # check if df is embedding or obj. if embedding, convert to obj
        df = self.df

        for col in df.columns:
            if df[col].dtype != "O":
                df = self.translate_embeddings(direction ="from")
                break

        if self.mode == "tail-batch":
            # get true targets and merge into dataframe
            target_df_tail = self.get_true_targets()
            df = df.drop(columns=["t"])
            df = df.merge(target_df_tail, on=["h", "r"], how="left")
            # fillna with empty list if empty
            df['t'].iloc[df[['t']].isnull().query('t==True').index] = df['t'].iloc[df[['t']].isnull().query('t==True').index].apply(lambda x: [])
            df["filt_preds"] = df.apply(
                lambda i: self.remove_list_from_list(i["preds"], i["t"]), axis=1
            )
            df = df.rename(columns={"t": "true_t"})
            df["filt_preds"] = df.apply(lambda i: self.remove_list_from_list(i["filt_preds"],i["h"]), axis=1)

        elif self.mode == "head-batch":
            target_df_head = self.get_true_targets()
            df = df.drop(columns=["h"])
            df = df.merge(target_df_head, on=["r", "t"], how="left")
            # fillna with empty list if empty
            df['h'].iloc[df[['h']].isnull().query('h==True').index] = df['h'].iloc[df[['h']].isnull().query('h==True').index].apply(lambda x: [])
            df["filt_preds"] = df.apply(
                lambda i: self.remove_list_from_list(i["preds"], i["h"]), axis=1
            )
            df = df.rename(columns={"h": "true_h"})
            df["filt_preds"] = df.apply(lambda i: self.remove_list_from_list(i["filt_preds"],i["t"]), axis=1)
            
        # get the top number of predictions
        df["filt_preds"] = df["filt_preds"].apply(lambda x: x[0:top])

        return df
    
    def calculate_position(self):
        """
        Get rank for each true prediction and return an n-hot vector of all hits

        Inputs
        -----------
        - self.df           * dataframe to calculate the position of each hit given 'trues'
        - self.mode         * {'head-batch','tail-batch'}
                              'head-batch': given (r,t) predict h
                              'tail-batch': given (h,r) predict t
        """
        df = self.df
        if self.mode == "tail-batch":
            # check if true values has been extracted, if not, extract and merge
            if "true_t" not in df.columns:
                target_df_tail = self.get_true_targets()
                df = df.rename(columns={"t":"target"})
                df = df.merge(target_df_tail, on=["h", "r"], how="left")
                df = df.rename(columns={"t": "true_t"})

            df["position"] = df.apply(
                lambda x: self.mark_list_from_list(x["preds"], x["true_t"]), axis=1
            )

        elif self.mode == "head-batch":
            # check if true values has been extracted, if not, extract and merge
            if "true_h" not in df.columns:
                target_df_head = self.get_true_targets()
                df = df.rename(columns={"h":"target"})
                df = df.merge(target_df_head, on=["r", "t"], how="left")
                df = df.rename(columns={"h": "true_h"})

            df["position"] = df.apply(
                lambda x: self.mark_list_from_list(x["preds"], x["true_h"]), axis=1
            )

        return df

    def calculate_mrr(self,show_position = False, show_rank = False, in_place = False) -> pd.DataFrame:
        """
        Calculates mean reciprocal rank and appends it to the dataframe
        Warning: Calculates the average reciprocal rank for all known targets. For just the current target, see calculate_individual_rr()
        
        Inputs
        -----------
        - self.df           * dataframe to calculate MRR from 'position'
        """
        df = self.df
        if "position" not in df.columns:
            df = self.calculate_position()

        # get index of all one's and get all one's preceding each index
        df["rank"] = df["position"].apply(lambda x: self.get_rank_from_position(x))

        # calculate mrr
        df["mrr"] = df["rank"].apply(lambda x: sum([1 / i for i in x]) / len(x))
        
        if show_position == False:
            df = df.drop(columns = ['position'])
        if show_rank == False:
            df = df.drop(columns = ['rank'])
        if in_place == True:
            self.df = df
        
        return df
    
    def calculate_individual_rank(self) -> pd.DataFrame:
        """
        Calculates the rank of the 'target' object in the 'predictions' and appends it to the dataframe as 'ind_rank'
        """
        
        df = self.df
        
        if "position" not in df.columns:
            df = self.calculate_position()

        # find the target, drop all 1's before the target.
        rank = list()
        for i, predictions in enumerate(df['preds']):
            target_position = self.find_target_position(df['target'][i],predictions)
            preceding_ind_count = sum(df['position'][i][0:target_position])

            target_rank = 1+target_position-preceding_ind_count
            rank.append(target_rank)

        df['ind_rank'] = rank #[1/i for i in rank]

        return(df)
    
    def calculate_individual_rr(self) -> pd.DataFrame:
        """
        Calculates the reciprocal rank of the 'target' object in the 'predictions' and appends it to the dataframe as 'rr'
        """
        
        df = self.df
        
        if "ind_rank" not in df.columns:
            df = self.calculate_individual_rank()
        
        df['rr'] = df['ind_rank'].apply(lambda x: 1/x)
        
        return(df)
    
    def calculate_hits_k(self, hits: list = [1,3,10], show_position = False, show_rank = False, in_place = False) -> pd.DataFrame:
        """
        Calculates the hits @ k given a list of k's
        Warning: Calculates the hit@k for all known targets. For just the current target, see calculate_individual_hits_k()
        
        Inputs
        -----------
        - self.df           * dataframe to calculate Hits at 'k' from 'position'
        - hits              * a list of ints to get the top "k" hits

        Returns columns of Hits at 'k' for each 'k' in list(hits)
        """
        df = self.df

        if "position" not in df.columns:
            df = self.calculate_position()
        if "rank" not in df.columns:
            df["rank"] = df["position"].apply(lambda x: self.get_rank_from_position(x))

        df["rank"] = df["rank"].apply(lambda x: np.array(x))
        for i in hits:
            df[f"hits_{i}"] = df["rank"].apply(lambda x: sum(x <= i) / len(x))

        if show_position == False:
            df = df.drop(columns = ['position'])
        if show_rank == False:
            df = df.drop(columns = ['rank'])
        if in_place == True:
            self.df = df
        return df
    
    def calculate_individual_hits_k(self, hits: list = [1,3,10], in_place = False) -> pd.DataFrame:
        """
        Calculates the hits@k given a list of k's. Default is hits @1, @3, and @10
        """
        df = self.df
        if "ind_rank" not in df.columns:
            df = self.calculate_individual_rank()
        
        for i in hits:
            df[f"hits_{i}"] = df["ind_rank"].apply(lambda x: x<=i)
        
        if in_place == True:
            self.df = df
        return df
    
    @staticmethod
    def find_target_position(target:str, prediction: list) -> int:
        """
        Finds the position of the 'target' string in a given list of strings 
        
        Example
        -----------
        target = 'abcd'
        list = ['ab','cd','ef','abcd','gh']
        
        find_target_position(target, list) -> 3 # index is zero
        """
        for i, val in enumerate(prediction):
            if val == target:
                return(i)
    
    @staticmethod
    def remove_list_from_list(x: list, y: list) -> list:
        """
        From list x, remove all entities that match in list y.

        Example
        ------------
        list(x) = [1,2,3,4,5]
        list(y) = [3,5,20]

        remove_list_from_list(x,y) -> [1,2,4]
        """    
        return [i for i in x if i not in y]

    @staticmethod
    def mark_list_from_list(x: list, y: list) -> list:
        """
        Takes list(x) and turns all entities not in list(y) to zero.
        Entities in list(y) in list(x) are turned to one.
        Returns list(x) as n-hot vector
        """
        return [0 if i not in y else 1 for i in x]

    @staticmethod
    def get_rank_from_position(x: list) -> np.array:
        """
        Gets positions of all hits and returns as an array

        Example
        ------------
        hits_list = [0,1,1,0,1,0,0,1]
        get_rank_from_position(hits_list) -> [2,2,4,7]
        """
        a = np.array(x)
        a = np.where(a == 1)
        a = a[0].tolist()
        b = [v - i + 1 for i, v in enumerate(a)]
        return b
    
    @property
    def df(self):
        '''dataframe property'''
        return self._df
    @df.setter
    def df(self, val):
        self._df = val