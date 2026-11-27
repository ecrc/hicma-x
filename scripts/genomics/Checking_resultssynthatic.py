import struct
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr,spearmanr

def from_bin(file_name, file_csv, rows, cols):
	with open(file_name, "rb") as file:
		data = file.read()
	n=[]
	n =struct.unpack("f"*(len(data) //4), data[:])
	with open("size.txt", "w") as output:
    		output.write(str(len(n)))
	a=np.array(n)
	a = a.reshape(rows, cols)
	df=pd.DataFrame(a)
	df.to_csv(file_csv, sep=',', header=False, index=False)
	return df

def check_mse_corr(array_1, array_2):
        try:
                m = mse(array_1, array_2)
        except Exception as e:
                m = None

        #r, _ = pearsonr(array_1, array_2)
        spr, _ = spearmanr(array_1, array_2)
        r=1
        return (m, r, spr)
def corr_row_wise(df1, df2, start, end, filename):
	r=list()
	for i in range(start, end):
		r.append(np.corrcoef(df1.loc[i, :].values, df2.loc[i, :].values))
	corr=[]
	with open(filename, "w") as f:
    		for s in r:
        		f.write(str(s[1][0]) +"\n")
        		corr.append(s[1][0])

	dff=pd.DataFrame(corr)
	print(dff.describe())

# for each dimension
def check_perf(truth, pred, dim):
        result_list_tmp = list()
        # check correlation coefficient
        # total mse
        # print('Total mse and corr')
        value_range = 'Dim_{}_Range_{}_{}'.format(dim, 0, 1)
        m, r, p = check_mse_corr(truth, pred)
        number = len(truth)
        result_list_tmp.append([value_range, number,
                m, r, p])


        # print('Breakdown mse and corr')
        # breakdown correlation coefficient
        # breakdown mse
        for i in range(10):
                # print('Range: {}, {}'.format(i*0.1, (i+1)*0.1))
                value_range = 'Dim_{}_Range_{}_{}'.format(dim, i*0.1, (i+1)*0.1)
                index = np.where((truth>i*0.1) & (truth<(i+1)*0.1))[0]
                # print('Number of samples: ',len(index))
                number = len(index)
                m, r, p = check_mse_corr(truth[index], pred[index])
                result_list_tmp.append([value_range, number,
                        m, r, p])
        return result_list_tmp

#ground_truth = from_bin('embedding.factor.permute.row_12740_x_columns_20.float8_min_max.bin', '1.csv', 12740, 20)
#ground_truth = pd.read_csv('embedding.factor.row_442719_x_columns_20.min_max.zero.removed.csv', header=None)
ground_truth = from_bin("/home/omairyrm/files/pheno10kx15_min.max_for_10kx10k.bin", "pheno10kx15_min.max_for_10kx10k.csv", 10000, 15)
predict=from_bin("YP_results.bin", "Y.csv", 15, 10000)

ground_truth = ground_truth.values
predict = pd.DataFrame(predict)
predict=predict.T
predict = predict.values
print(predict.shape)
train_ground_truth = ground_truth[:int(len(predict)*0.8), :]
test_ground_truth = ground_truth[int(len(predict)*0.8):len(predict), :]

all_ground_truth = ground_truth[:len(predict), :]

train_predict = predict[:int(len(predict)*0.8), :]
test_predict = predict[int(len(predict)*0.8):len(predict), :]
all_predict = predict[:len(predict), :]

result_list_la = list()
for i in range(15):
        result_list_tmp = check_perf(test_ground_truth[:, i], test_predict[:, i], i)
        result_list_la += result_list_tmp

result_df_la = pd.DataFrame(result_list_la, columns=['value_range', 'number',
        'MSE_LA', 'pearsonr_LA', 'spearmanr_LA'])
print(result_df_la.iloc[np.arange(15)*11].describe())
result_df_la.to_csv('LA_result.csv')

print(result_df_la.iloc[np.arange(15)*11])
print('Here is the MSE of the training data: ', mse(train_ground_truth, train_predict))
print('Here is the MSE of the testing data: ', mse(test_ground_truth, test_predict))
print('Here is the MSE of all subset data: ', mse(all_ground_truth, all_predict))

corr_row_wise(pd.DataFrame(ground_truth), pd.DataFrame(predict), int(len(predict)*0.8), len(predict), "corr-row.csv")
