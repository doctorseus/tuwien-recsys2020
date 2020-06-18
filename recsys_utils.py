import pandas as pd
import numpy as np

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
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)
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
  
def recsys_evaluate(df_data, recommender_train_predict_cb, type='all'):
    df_results = pd.DataFrame()

    print('Run RecSys recommender evaluation:')
    
    for split_type, split_func in zip(['single_random', 'tweetid', 'userid', 'time'], [recsys_cv_split_single, recsys_cv_split_tweetid, recsys_cv_split_userid, recsys_cv_split_time]):
        if type == 'all' or type == split_type or split_type in type:
            print('> cv-split (%s)' % split_type)

            results_prauc = []
            for df_train, df_test in split_func(df_data):
                gt_reply, gt_retweet, gt_retweet_wc, gt_like = get_test_gt(df_test)

                pred_reply, pred_retweet, pred_retweet_wc, pred_like = zip(*list(recommender_train_predict_cb(df_train, df_test)))


                prauc_reply = compute_prauc(pred_reply, gt_reply)
                prauc_retweet = compute_prauc(pred_retweet, gt_retweet)
                prauc_retweet_wc = compute_prauc(pred_retweet_wc, gt_retweet_wc)
                prauc_like = compute_prauc(pred_like, gt_like)

                results_prauc.append([prauc_reply, prauc_retweet, prauc_retweet_wc, prauc_like])

                print('.', end='')
            print()

            results_prauc_mean = np.array(results_prauc).mean(axis=0)
            df_results = df_results.append({
                'split_type': split_type,
                'PRAUC(reply)': results_prauc_mean[0],
                'PRAUC(retweet)': results_prauc_mean[1],
                'PRAUC(retweet_wc)': results_prauc_mean[2],
                'PRAUC(like)': results_prauc_mean[3]
            }, ignore_index=True)

    df_results = df_results.set_index('split_type')
    return df_results
