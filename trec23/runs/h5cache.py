import numpy as np
import more_itertools
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pyterrier as pt
from npids import Lookup
import ir_datasets
import shutil
logger = ir_datasets.log.easy()

class H5CacheScorer(pt.transformer.TransformerBase):
    def __init__(self, path, scorer, force=False):
        self.mode = 'r'
        self.path = Path(path)
        assert (self.path/'meta.json').exists()
        with (self.path/'meta.json').open('rt') as fin:
            self.meta = json.load(fin)
        assert self.meta.get('type') == 'score_cache' and self.meta.get('format') == 'h5'
        self.file = None
        self.scorer = scorer
        self.corpus_count = self.meta['doc_count']
        self.dataset_cache = {}
        self.docnos = Lookup(self.path/'docnos.npids')
        self.force = force
        
    def transform(self, inp):
        results = []
        inp.reset_index()
        if self.force:
            inp = self.scorer(inp)
        for query, group in inp.groupby('query'):
            ds = self.get_dataset(query)
            dids = self.docnos.inv[group.docno]
            dids, unique_inv = np.unique(dids, return_inverse=True)
            scores = ds[dids][unique_inv]
            to_score = group.loc[group.docno[np.isnan(scores)].index]
            if len(to_score) > 0:
                self.ensure_write_mode()
                ds = self.get_dataset(query)
                if self.force:
                    new_scores = to_score
                else:
                    new_scores = self.scorer(to_score)
                scores_by_docno = dict(zip(new_scores['docno'], new_scores['score']))
                docnos = np.array(list(scores_by_docno.keys()))
                scored_dids = self.docnos.inv[docnos]
                scored_did_sort = scored_dids.argsort()
                ds[scored_dids[scored_did_sort]] = [scores_by_docno[d] for d in docnos[scored_did_sort]]
                scores = ds[dids][unique_inv]
            group = group.assign(score=scores)
            group.sort_values('score', ascending=False, inplace=True)
            results.append(group)
        results = pd.concat(results, ignore_index=True)
        pt.model.add_ranks(results)
        return results
        
    def ensure_write_mode(self):
        if self.mode == 'r':
            import h5py
            self.mode = 'a'
            self.file.close()
            self.file = h5py.File(self.path/'data.h5', self.mode)
            self.dataset_cache = {} # file changed, need to reset the cache
            
    def get_dataset(self, query):
        import h5py
        if self.file is None:
            self.file = h5py.File(self.path/'data.h5', self.mode)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        if query_hash not in self.dataset_cache:
            if query_hash not in self.file:
                self.ensure_write_mode()
                self.file.create_dataset(query_hash, shape=(self.corpus_count,), dtype=np.float32, fillvalue=float('nan'))
            self.dataset_cache[query_hash] = self.file[query_hash]
        return self.dataset_cache[query_hash]
        
    @staticmethod
    def create_from_dataset(dataset, path, scorer, init_docnos=None, force=False):
        import h5py
        path = Path(path)
        assert not path.exists()
        path.mkdir(parents=True)
        file = h5py.File(str(path/'data.h5'), 'a')
        if init_docnos:
            with logger.duration(f'copying docnos from {init_docnos}'):
                shutil.copy(init_docnos, path/'docnos.npids')
            docnos = Lookup(path/'docnos.npids')
        else:
            docnos = Lookup.build((d.doc_id for d in logger.pbar(dataset.irds_ref().docs, desc='building docno lookup', unit='doc')), path/'docnos.npid')
        with (path/'meta.json').open('wt') as fout:
            json.dump({
                'type': 'score_cache',
                'format': 'h5',
                'doc_count': len(docnos),
            }, fout)
        return H5CacheScorer(path, scorer, force=force)
        
    @staticmethod
    def init_or_create(dataset, path, scorer, force=False):
        if (Path(path)/'meta.json').exists():
            return H5CacheScorer(path, scorer, force=force)
        return H5CacheScorer.create_from_dataset(dataset, path, scorer, force=force)
        
    def full_scorer(self):
        return FullScorer(self)
        
class FullScorer(pt.Transformer):
    def __init__(self, cache):
        self.cache = cache
    def transform(self, inp):
        results = {'qid': [], 'query': [], 'docid': [], 'score': [], 'rank': []}
        inp.reset_index()
        for query, group in inp.groupby('query'):
            ds = self.cache.get_dataset(query)
            dids = np.argpartition(ds, -1000)[-1000:]
            dids.sort()
            scores = ds[dids]
            idxs = (-scores).argsort()
            dids = dids[idxs]
            scores = scores[idxs]
            results['qid'].extend([group.iloc[0]['qid']] * 1000)
            results['query'].extend([query] * 1000)
            results['docid'].append(dids)
            results['score'].append(scores)
            results['rank'].append(np.arange(1000))
        results['docid'] = np.concatenate(results['docid'])
        results['docno'] = self.cache.docnos.fwd[results['docid']]
        results['score'] = np.concatenate(results['score'])
        results['rank'] = np.concatenate(results['rank'])
        return pd.DataFrame(results)
        
class ExhaustiveSearch(pt.Transformer):
    def __init__(self, scorer, dataset):
        self.scorer = scorer
        self.dataset = dataset
        
    def transform(self, inp):
        inp['key'] = 0
        prev = None
        for batch in more_itertools.chunked(self.dataset.get_corpus_iter(), 10_000):
            batch = pd.DataFrame(batch)
            batch['key'] = 0
            batch = pd.merge(inp, batch, on='key', how='outer')
            this = self.scorer(batch)
            if prev is None:
                prev = this
            else:
                prev = pd.concat([prev, this])
                pt.model.add_ranks(prev)
                prev = prev[prev['rank'] < 1000]
        return prev