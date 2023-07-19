import pyterrier as pt
import pandas as pd
from pyterrier.model import add_ranks
from copy import deepcopy

class MarcoDuplicator(pt.transformer):
    essential_metadata = ['docno']
    def __init__(self, lookup, **kwargs):
        self.lookup = lookup
        super().__init__(**kwargs)

    def insert_row(self, idx, df, df_insert):
        return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)

    def transform(self, input : pd.DataFrame):
        assert self.essential_metadata.issubset(input.columns), f"input must contain {self.essential_metadata}"
        input = input.copy()

        changes = []
        for row in input.itertuples():
            if row.docno in self.lookup:
                tmp = []
                for id in self.lookup[row.docno]:
                    tmp_row = deepcopy(row)
                    setattr(tmp_row, 'docno', id)
                    tmp.append(tmp_row)
                changes.append(pd.DataFrame(tmp))
        
        changes = pd.concat(changes, ignore_index=True).reset_index(drop=True)
        input = pd.concat([input, changes], ignore_index=True).reset_index(drop=True)

        input.drop(['rank'], axis=1, inplace=True)
        input = add_ranks(input)

        return input

        