import pyterrier as pt
import pandas as pd

class DuplicatorTransformer(pt.transformer):
    essential_metadata = ['docno']
    def __init__(self, lookup, **kwargs):
        self.lookup = lookup
        super().__init__(**kwargs)

    def insert_row(self, idx, df, df_insert):
        return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)

    def transform(self, input : pd.DataFrame):
        assert self.essential_metadata.issubset(input.columns), f"input must contain {self.essential_metadata}"
        input = input.copy().reset_index(drop=True)
        changes = {}
        for row in input.itertuples():
            if row.docno in self.lookup:
                tmp = []
                for id in self.lookup[row.docno]:
                    tmp_row = row
                    setattr(tmp_row, 'docno', id)
                    tmp.append(tmp_row)
                changes[row.docno] = pd.DataFrame(tmp)

        for docno, df_insert in changes.items():      
            input = self.insert_row(input.index[input['docno'] == docno].tolist()[0], input, df_insert)

        