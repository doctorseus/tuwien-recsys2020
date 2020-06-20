import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, Manager

from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, auc, log_loss
from sklearn.model_selection import train_test_split

column_names = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",\
               "engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
               
cv_split_folds = 10

def recsys_load_training_df(file_path):
    df = pd.read_csv(file_path, header=None, names=column_names, delimiter='\x01')
    
    df['text_tokens'] = df['text_tokens'].str.split('\t')

    def to_hex_list(x):
        output = str(x).split('\t')
        return output

    cols_to_process = ['hashtags', 'present_media', 'present_links', 'present_domains']

    for col in cols_to_process:  
        df[col] = df[col].apply(lambda x: to_hex_list(x) if isinstance(x, str)  else x)


    cols_to_process = ['tweet_timestamp', 'engaging_user_account_creation', 'reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

    for col in cols_to_process:  
        df[col] = df[col].apply(lambda x: pd.Timestamp(x, unit='s'))
    
    return df

def recsys_cv_split_single(df):
    user_counts = df['engaging_user_id'].value_counts()
    df_filtered = df[~df['engaging_user_id'].isin(user_counts[user_counts < 2].index)]
    
    df_train, df_test = train_test_split(df_filtered, stratify=df_filtered['engaging_user_id'], test_size=0.20, random_state=42)
    return [(df_train, df_test)]

def recsys_cv_split_tweetid(df):
    unique_tweet_ids = df['tweet_id'].unique()
    unique_tweet_ids.sort()
    n = len(unique_tweet_ids)

    tweetId_to_tweetIDX = dict(zip(unique_tweet_ids, range(n)))
    tweetIDX_to_tweetId = dict(zip(range(n), unique_tweet_ids))
    
    cv = KFold(n_splits=cv_split_folds, shuffle=True)

    for train_idx, dev_idx in cv.split(unique_tweet_ids):
        train_df = df.loc[df.tweet_id.isin(map(tweetIDX_to_tweetId.get, train_idx)),:]
        dev_df = df.loc[df.tweet_id.isin(map(tweetIDX_to_tweetId.get, dev_idx)),:]
        yield train_df, dev_df

def recsys_cv_split_userid(df):
    unique_user_ids = df['engaging_user_id'].unique()
    unique_user_ids.sort()
    m = len(unique_user_ids)

    userId_to_userIDX = dict(zip(unique_user_ids, range(m)))
    userIDX_to_userId = dict(zip(range(m), unique_user_ids))

    cv = KFold(n_splits=cv_split_folds, shuffle=True)

    for train_idx, dev_idx in cv.split(unique_user_ids):
        train_df = df.loc[df.engaging_user_id.isin(map(userIDX_to_userId.get, train_idx)),:]
        dev_df = df.loc[df.engaging_user_id.isin(map(userIDX_to_userId.get, dev_idx)),:]
        yield train_df, dev_df

def recsys_cv_split_time(df):
    cv = TimeSeriesSplit(n_splits=cv_split_folds)
    df_sorted = df.sort_values(by=['tweet_timestamp'])
    for ii, (train_idx, dev_idx) in enumerate(cv.split(df_sorted)):
        train_df = df_sorted.loc[train_idx,:]
        dev_df = df_sorted.loc[dev_idx,:]
        yield train_df, dev_df

def recsys_cv_split_time_single(df):
    cv = TimeSeriesSplit(n_splits=cv_split_folds)
    df_sorted = df.sort_values(by=['tweet_timestamp'])
    time_splits = list(cv.split(df_sorted))

    train_idx, dev_idx = time_splits[len(time_splits) - 1]
    train_df = df_sorted.loc[train_idx,:]
    dev_df = df_sorted.loc[dev_idx,:]
    return [(train_df, dev_df)]
    

def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def get_test_gt(df_test):
    return (~df_test['reply_timestamp'].isna()).astype(int).to_numpy(), (~df_test['retweet_timestamp'].isna()).astype(int).to_numpy(), (~df_test['retweet_with_comment_timestamp'].isna()).astype(int).to_numpy(), (~df_test['like_timestamp'].isna()).astype(int).to_numpy()

def recsys_evaluation_worker(result_queue, split_type, recommender_train_predict_cb, df_rain, df_test):
    try:
        pred_reply, pred_retweet, pred_retweet_wc, pred_like = zip(*list(recommender_train_predict_cb(df_rain, df_test)))

        gt_reply, gt_retweet, gt_retweet_wc, gt_like = get_test_gt(df_test)
        result_queue.put({
            'split_type': split_type,
            'reply (PRAUC)': compute_prauc(pred_reply, gt_reply),
            'reply (CTR)': calculate_ctr(gt_reply),
            'reply (RCE)': compute_rce(pred_reply, gt_reply),

            'retweet (PRAUC)': compute_prauc(pred_retweet, gt_retweet),
            'retweet (CTR)': calculate_ctr(gt_retweet),
            'retweet (RCE)': compute_rce(pred_retweet, gt_retweet),

            'retweet_wc (PRAUC)': compute_prauc(pred_retweet_wc, gt_retweet_wc),
            'retweet_wc (CTR)': calculate_ctr(gt_retweet_wc),
            'retweet_wc (RCE)': compute_rce(pred_retweet_wc, gt_retweet_wc),

            'like (PRAUC)': compute_prauc(pred_like, gt_like),
            'like (CTR)': calculate_ctr(gt_like),
            'like (RCE)': compute_rce(pred_like, gt_like),
        })
    except Exception as e:
        print('Evaluation failed: %s' % e)

def recsys_evaluate(df_data, recommender_train_predict_cb, type='all', parallel=True):
    print('Run RecSys recommender evaluation:')
    cpus = cpu_count() if parallel else 1
    print(f'CPUs: {cpus}')

    pool = Pool(int(cpus))
    m = Manager()
    result_queue = m.Queue()
    
    for split_type, split_func in zip(['single_random', 'tweetid', 'userid', 'time', 'single_time'], [recsys_cv_split_single, recsys_cv_split_tweetid, recsys_cv_split_userid, recsys_cv_split_time, recsys_cv_split_time_single]):
        if type == 'all' or type == split_type or (isinstance(type, list) and split_type in type):
            print('> cv-split (%s)' % split_type)
            for df_train, df_test in split_func(df_data):
                pool.apply_async(recsys_evaluation_worker, (result_queue, split_type, recommender_train_predict_cb, df_train.reset_index(drop=True), df_test.reset_index(drop=True)))

    pool.close()
    pool.join()

    # Collect results
    df_results = pd.DataFrame()
    while not result_queue.empty():
        result = result_queue.get()
        df_results = df_results.append(result, ignore_index=True)

    df_results_mean = df_results.groupby('split_type').mean()
    df_results_mean['agg'] = 'mean'

    df_results_min = df_results.groupby('split_type').min()
    df_results_min['agg'] = 'min'

    df_results_max = df_results.groupby('split_type').max()
    df_results_max['agg'] = 'max'
    return pd.concat([df_results_mean, df_results_min, df_results_max]).set_index(['agg'], append=True).sort_index()
