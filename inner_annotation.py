from utils import read_tsv_file
from copy import deepcopy
from ner_evaluation import collect_named_entities
from ner_evaluation import compute_metrics
#Annotator 1 versus Annotator 2
#data_true,labels_true = read_tsv_file('./data/anno_one.tsv', keep_conflict=True)
#data_pred,labels_pred = read_tsv_file('./data/anno_two.tsv', keep_conflict=True)

#Annotator 1 versus Annotator 3
#data_true,labels_true = read_tsv_file('./data/anno_one.tsv', keep_conflict=True)
#data_pred,labels_pred = read_tsv_file('./data/anno_three.tsv', keep_conflict=True)

#Annotator 1 versus Annotator 3
data_true,labels_true = read_tsv_file('./data/anno_two.tsv', keep_conflict=True)
data_pred,labels_pred = read_tsv_file('./data/anno_three.tsv', keep_conflict=True)
#change other

metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                   'missed': 0, 'spurius': 0, 'possible': 0, 'actual': 0}

# overall results
results = {'strict': deepcopy(metrics_results),
           'ent_type': deepcopy(metrics_results)
           }

# results aggregated by entity type
evaluation_agg_entities_type = {e: deepcopy(results) for e in ['NEU','NEG','POS']}

for true_ents, pred_ents in zip(labels_true, labels_pred):
    # compute results for one message
    tmp_results, tmp_agg_results = compute_metrics(collect_named_entities(true_ents),collect_named_entities(pred_ents))

    # aggregate overall results
    for eval_schema in results.keys():
        for metric in metrics_results.keys():
            results[eval_schema][metric] += tmp_results[eval_schema][metric]
        correct = results[eval_schema]['correct']
        partial = results[eval_schema]['partial']
        actual = results[eval_schema]['actual']
        possible = results[eval_schema]['possible']
        if eval_schema == 'ent_type':#partial_matching
            results[eval_schema]['precision'] = (correct + 0.5 * partial) / actual if actual > 0 else 0
            results[eval_schema]['recall'] = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else: #strict_matching
            results[eval_schema]['precision'] = correct / actual if actual > 0 else 0
            results[eval_schema]['recall'] = correct / possible if possible > 0 else 0
print(results)

