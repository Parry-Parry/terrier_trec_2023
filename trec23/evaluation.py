import pyterrier as pt

def generate_experiment(*models, dataset=None, **kwargs):
    if not pt.started(): 
        pt.init()
    if not dataset:
        topics = kwargs.get("topics", None)
        qrels = kwargs.get("qrels", None)
        assert topics is not None and qrels is not None, "Topics and Qrels must be specified if Dataset is not"
    else:
        topics = dataset.get_topics()
        qrels = dataset.get_qrels()

    args = {
        "retr_systems" : list(models),
        "topics" : topics,
        "qrels" : qrels,
        "eval_metrics" : kwargs.get("metrics", ["map", "ndcg_cut_10", "mrr_cut_10"]),
        "names" : kwargs.get("names", [f'model_{i}' for i in range(len(models))]),
        "perquery" : kwargs.get("per_query", False),
        "batch_size" : kwargs.get("batch_size", None),
        "save_dir" : kwargs.get("save_dir", None),
        "baseline" : kwargs.get("baseline", None),
        "test" : kwargs.get("test", None),
        "correction" : kwargs.get("correction", None),
        "verbose" : kwargs.get("verbose", False)
    }

    return pt.Experiment(**args)

def dual_experiment(*models, dataset=None, **kwargs):
    if not pt.started(): 
        pt.init()
    if not dataset:
        topics = kwargs.get("topics", None)
        qrels = kwargs.get("qrels", None)
        assert topics is not None and qrels is not None, "Topics and Qrels must be specified if Dataset is not"
    else:
        topics = dataset.get_topics()
        qrels = dataset.get_qrels()

    args = {
        "retr_systems" : list(models),
        "topics" : topics,
        "qrels" : qrels,
        "eval_metrics" : kwargs.get("metrics", ["map", "ndcg_cut_10", "mrr_cut_10"]),
        "names" : kwargs.get("names", [f'model_{i}' for i in range(len(models))]),
        "batch_size" : kwargs.get("batch_size", None),
        "save_dir" : kwargs.get("save_dir", None),
        "verbose" : kwargs.get("verbose", False)
    }

    std_args = args.copy()
    std_args["perquery"] = False
    std_args["baseline"] = kwargs.get("baseline", None)
    std_args["test"] = kwargs.get("test", None)
    std_args["correction"] = kwargs.get("correction", None)

    perquery_args = args.copy()
    perquery_args["perquery"] = True
    perquery_args["baseline"] = None 
    perquery_args["test"] = None
    perquery_args["correction"] = None

    return pt.Experiment(**std_args), pt.Experiment(**perquery_args)
