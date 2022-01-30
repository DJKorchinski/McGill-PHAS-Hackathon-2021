import pickle 
import CalcBFromS

#takes the output of "grid_search" and computes binders cumulants for all the data points. 

src = 'data/cluster_size.npy'
# with open(src,'rb') as fh:
#     dic = pickle.load(fh)

import numpy as np 

dic = np.load(src,allow_pickle=True).tolist()

pm = dic['param_manager']
sizes = dic['cluster_sizes']
params = []
cluster_sizes_sorted = []

for repno in range(pm.REP_TOTAL):
    rep_params = pm.get_params(repno)
    rep_param_subset = [rep_params[i] for i in range(3)]
    
    #check to see if 
    found = False 
    for j,param_check in enumerate(params):
        if(rep_param_subset == param_check):
            found = True 
            break
    if(found):
        cluster_sizes_sorted[j].append(sizes[repno])
    else:
        params.append(rep_param_subset)
        cluster_sizes_sorted.append([sizes[repno]])
    

cluster_sizes_sorted = [[np.array(entry) for entry in lst] for lst in cluster_sizes_sorted]

#now that the data is sorted, run the binder cumulant on it!
binders = []
for i,param_set in enumerate(params):
    binders.append( CalcBFromS.B(cluster_sizes_sorted[i],param_set[2]))

#save the processed data:
import pickle 
output_dic = {'params':params,'cluster_sizes_sorted':cluster_sizes_sorted,'binders':binders}
output_fname = 'data/analysis_binders_01.dat'
with open(output_fname,'wb') as fh:
    pickle.dump(output_dic,fh)

#make a nice text file for Raelyn
binders_output = np.array([[param[0],param[1],param[2],binders[i]] for i,param in enumerate(params)])
np.save('data/grid_search_binders.npy',binders_output)