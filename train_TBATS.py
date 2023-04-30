import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tbats import TBATS
from sklearn.metrics import mean_absolute_percentage_error as mape

## load data
dat_pred = pd.read_csv('dat_pred.csv')
dat_pred.set_index('label',inplace=True)



def train_TBATS(data,train_len=672, step=168):
	print('*'*20,'training TBATS models','*'*20)
	pred_tb_all = pd.DataFrame()    # store prediction results for 5 clusters
	mape_list = []   # store mape for 5 clusters
	# train_len = 24 * 7 * 4
	# step = 24 * 7    # one week ahead prediction
	# step = 24    # one day ahead prediction (higher accuracy)
	num_batch = int(np.ceil((dat_pred.shape[1] - train_len) / step))
	num_clus = dat_pred.shape[0]

	for k in range(num_clus):
		print('*' * 20, 'cluster', k, '*' * 20)
		pred_k_tb = pd.DataFrame()  # store forecast data

		for i in range(num_batch):  # for each training window
			print('training window: ', i)

			# prepare train and test data
			train_idx = np.arange(step * i, train_len + step * i)
			test_idx = np.arange(train_len + step * i,
			                     min(train_len + step * (i + 1), dat_pred.shape[1]))  # use min to handle the last week
			train = dat_pred.iloc[k, train_idx]
			test = dat_pred.iloc[k, test_idx]

			# TBATS
			# daily and weekly seasonality
			estimator = TBATS(seasonal_periods=[24,168], use_arma_errors=False, use_box_cox=False)
			fitted_model = estimator.fit(train)
			pred_tmp = fitted_model.forecast(steps=min(step, len(test)))
			pred_tmp = pd.DataFrame(pred_tmp, index=test.index)
			pred_k_tb = pd.concat([pred_k_tb, pred_tmp])
		pred_tb_all = pd.concat([pred_tb_all,pred_k_tb],axis=1)

		# calculate MAPE of cluster k
		true_k = dat_pred.iloc[k, train_len:]
		mape_tb = mape(true_k,pred_k_tb)
		mape_list.append(mape_tb)
		print( 'MAPE for cluster {}: {}'.format(k,mape_tb))
		print( 'accuracy: {:.2f}%'.format((1-mape_tb)*100))
	print('overall accuray: {:.2f}%'.format((1-np.mean(mape_list))*100))
	return {'result':pred_tb_all, 'mape':mape_list}

# pred_tb_all.to_csv('pred_results_TB_1w.csv')    # overall accuray: 80.20%
# pred_tb_all.to_csv('pred_results_TB_1d.csv')    # overall accuray: 83.84%



if __name__ == "__main__":
	result_tb, mape_tb = train_TBATS(dat_pred, train_len=672, step=168)
	result_tb.to_csv('pred_results_TB_1d.csv')
