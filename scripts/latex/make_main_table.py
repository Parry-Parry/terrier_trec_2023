import pandas as pd 
from fire import Fire

def main(file : str, out : str): 
    BASELINE = [
        'uog_tr_dph',
        'uog_tr_s',
        'uog_tr_dph_bo1',
        'uog_tr_be',
        'uog_tr_se'
    ]
    MAIN = [
        'uog_tr_se_gb',
        'uog_tr_qr_be_gb',
        'uog_tr_b_grf_e_gb',
        'uog_tr_qr_be', 
        'uog_tr_b_grf_e', 
        'uog_tr_be_gb'
    ]

    column2metric = {
        'P_10' : 'P@10',
        'ndcg_cut_10' : 'NDCG@10',
        'recip_rank' : 'MRR', 
        'map' : 'MAP',
        'recall_100' : 'R@100'
        }

    metric2column = {
        'P@10' : 'P_10',
        'NDCG@10' : 'ndcg_cut_10', 
        'MRR' : 'recip_rank', 
        'MAP' : 'map',
        'R@100' : 'recall_100'
        }

    BEST = {
        'P_10' : 0.7000,
        'ndcg_cut_10' : 0.7892,
        'recip_rank' : 0.9939,
        'map' : 0.3839, 
        'recall_100' : 1.
    }
    MEDIAN = {
        'P_10' : 0.4085,
        'ndcg_cut_10' : 0.5329,
        'recip_rank' : 0.7803,
        'map' : 0.2159,
        'recall_100' : 1.
    }

    COLOURS = {
        'P_10' : 'buwyellow',
        'ndcg_cut_10' : 'buwblue',
        'recip_rank' : 'buwgreen',
        'map' : 'buworange', 
        'recall_100' : 'buwred'
    }

    def format_metric(value, colour_token, colour_level, better):
        better_tok = r'\better' if better else r'\worse'
        #out = r'\cellcolor{' + colour_token + f'!{colour_level}' + '}' + f'${abs(value)}{better_tok}$' 
        out = f'${abs(value)}{better_tok}$'
        return out

    def colour_combo(metric, value, control=False):
            max_val = float(BEST[metric])
            median_val = float(MEDIAN[metric])
            abs_val = abs(value)
            # min max normalise abs_val between max val and 0 
            norm_val = (abs_val - 0) / (max_val - 0)
            norm_val = round(norm_val * 50)
            norm_val = min(norm_val, 50)
            return format_metric(round(value, 4), COLOURS[metric], norm_val, abs_val > median_val and not control)

    df = pd.read_csv(file, index_col=False)
    # melt such that we have a column for each metric

    df = df.melt(id_vars=['runid'], value_vars=['P_10', 'ndcg_cut_10', 'recip_rank', 'map'], var_name='metric', value_name='value')
    # print dtypes 
    print(df.runid.unique())

    preamble = [r"\begin{table*}", r"\centering", r"\begin{tabular}{@{}ll|cccc@{}}", r"\toprule"]
    header = r"Run ID & Pipeline & P@10 & NDCG@10 & MRR & MAP \\"

    best_median = []

    best_line = r'\multicolumn{2}{l|}{Best (Per-Topic)} & &'
    median_line = r'\multicolumn{2}{l|}{Median (Per-Topic)} & &'

    for metric, column in metric2column.items():
        print(BEST[column])
        best_line += colour_combo(column, BEST[column], True) + ' & '
        median_line += colour_combo(column, MEDIAN[column], True) + ' & '
    
    best_line = best_line[:-2] + r'\\'
    median_line = median_line[:-2] + r'\\'
    

    output = [*preamble, header, r'\midrule', best_line, median_line, r'\midrule',  r'\multicolumn{6}{l}{' + 'Baseline Runs' + r'}\\', r'\midrule']
    '''
    for model in BASELINE: 
        subset = df.loc[df['runid'] == str(model)].copy()
        print(subset)
        line = f'{model} & &'
        for metric, column in metric2column.items():
            value = subset[subset.metric==column].values[0]
            line += colour_combo(metric, value) + ' &'
        line = line[:-1] + r'\\'
        output.append(line)
    
    output.extend([r'\midrule', r'\multicolumn{5}{l}{' + 'Main Runs' + r'}\\', r'\midrule'])
      
    for model in MAIN:
        subset = df.loc[df['runid'] == model].copy()
        line = f'{model} & &'
        for metric, column in metric2column.items():
            value = subset[subset.metric==column].values[0]
            line += colour_combo(metric, value) + ' &'
        line = line[:-1] + r'\\'        
        output.append(line)
    '''

    for model in df.runid.unique():
        subset = df.loc[df['runid'] == model].copy()
        runid = model.replace('_', r'\_')
        line = f'{runid} & &'
        for metric, column in metric2column.items():
            try: 
                value = subset[subset.metric==column].value.values[0]
            except IndexError:
                value = 100.
            line += colour_combo(column, value) + ' & '
        line = line[:-2] + r'\\'        
        output.append(line)

    
    output.extend([r'\bottomrule', r'\end{tabular}', r'\caption{Main Table}', r'\label{tab:main}', r'\end{table*}'])

    with open(out, 'w') as f:
        f.write('\n'.join(output))

if __name__ == '__main__':
    Fire(main)