import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import Adam

import xgboost as xgb

import seaborn as sns
from scipy import stats
import pyensae
from pyensae.graphhelper import Corrplot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
#import graphviz 
import os,shutil
import random

import time
import traceback
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

folder_GBM = 'Est_GBM'
if os.path.exists(folder_GBM):
    shutil.rmtree(folder_GBM)
os.makedirs(folder_GBM)

folder_XGB = 'Est_XGB'
if os.path.exists(folder_XGB):
    shutil.rmtree(folder_XGB)
os.makedirs(folder_XGB)

T=100
nTrees = 100
lRate = 0.05
mDepth = 5

fsize = (12,8)

inden_vector_skmp = ['#Pdb', 'Mutation(s)_cleaned', 'iMutation_Location(s)', 'Hold_out_type', 'Method', 'Affinity_wt', 'Affinity_mut']
inden_vector_skmp_wt = ['#Pdb', 'Hold_out_type', 'Method', 'Affinity_wt']
inden_vector_affB = ['Complex PDB', 'Type', 'Method', 'dG']
#score_vector_lisa = ['V106','V46','V202','V208','V107','V114','V154','V207','V69','nis2','b_affine']
score_vector_lisa = ['V106','V46','V202','V208','V107','V114','V154','V207','V69','nis2']
"""
score_vector_lisa_original = ["V39","V40","V41","V42","V43","V44","V45","V46","V47","V48","V49","V50","V51","V52","V53","V54",
                              "V55","V56","V57","V58","V59","V60","V61","V62","V63","V64","V65","V66","V67","V68","V69","V70",
                              "V71","V72","V73","V74","V75","V76","V77","V78","V79","V80","V81","V82","V83","V84","V85","V86",
                              "V87","V88","V89","V90","V91","V92","V93","V94","V95","V96","V97","V98","V99","V100","V101","V102",
                              "V103","V104","V105","V106","V107","V108","V109","V110","V111","V112","V113","V114","V115","V116",
                              "V117","V118","V119","V120","V121","V122","V123","V124","V125","V126","V127","V128","V129","V130","V131",
                              "V132","V133","V134","V135","V136","V137","V138","V139","V140","V141","V142","V143","V144","V145","V146",
                              "V147","V148","V149","V150","V151","V152","V153","V154","V155","V156","V157","V158","V159","V160","V161",
                              "V162","V163","V164","V165","V166","V167","V168","V169","V170","V171","V172","V173","V174","V175","V176",
                              "V177","V178","V179","V180","V181","V182","V183","V184","V185","V186","V187","V188","V189","V190","V191",
                              "V192","V193","V194","V195","V196","V197","V198","V199","V200","V201","V202","V203","V204","V205","V206",
                              "V207","V208","V209","V210","V211","V212","V213","V214","nis1","nis2","nis3","IntVol1","IntVol2","IntVol3",
                              "IntVol4","IntVol5","IntVol6","IntVol7","IntVol8","IntVol9","IntArea1","IntArea2","IntArea3","IntArea4","IntArea5",
                              "IntArea6","IntArea7","IntArea8","IntArea9","LogIntVol1","LogIntVol2","LogIntVol3","LogIntVol4","LogIntVol5","LogIntVol6",
                              "LogIntVol7","LogIntVol8","LogIntVol9","LogIntArea1","LogIntArea2","LogIntArea3","LogIntArea4","LogIntArea5","LogIntArea6",
                              "LogIntArea7","LogIntArea8","LogIntArea9","LogNumCon"]
"""
score_vector_lisa_original = ["V39","V40","V41","V42","V43","V44","V45","V46","V47","V48","V49","V50","V51","V52","V53","V54",
                              "V55","V56","V57","V58","V59","V60","V61","V62","V63","V64","V65","V66","V67","V68","V69","V70",
                              "V71","V72","V73","V74","V75","V76","V77","V78","V79","V80","V81","V82","V83","V84","V85","V86",
                              "V87","V88","V89","V90","V91","V92","V93","V94","V95","V96","V97","V98","V99","V100","V101","V102",
                              "V103","V104","V105","V106","V107","V108","V109","V110","V111","V112","V113","V114","V115","V116",
                              "V117","V118","V119","V120","V121","V122","V123","V124","V125","V126","V127","V128","V129","V130","V131",
                              "V132","V133","V134","V135","V136","V137","V138","V139","V140","V141","V142","V143","V144","V145","V146",
                              "V147","V148","V149","V150","V151","V152","V153","V154","V155","V156","V157","V158","V159","V160","V161",
                              "V162","V163","V164","V165","V166","V167","V168","V169","V170","V171","V172","V173","V174","V175","V176",
                              "V177","V178","V179","V180","V181","V182","V183","V184","V185","V186","V187","V188","V189","V190","V191",
                              "V192","V193","V194","V195","V196","V197","V198","V199","V200","V201","V202","V203","V204","V205","V206",
                              "V207","V208","V209","V210","V211","V212","V213","V214","nis1","nis2","nis3"]



score_vector_lisa_original_groups = {'lg1': score_vector_lisa_original[:168], 
                                     'lg2': score_vector_lisa_original[168:176], 
                                     'lg3': score_vector_lisa_original[176:]}


score_vector_flex = ['fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_rep','fa_sol','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','nstruct','omega','p_aa_pp','pro_close','rama','ref','total_score','yhh_planarity'] 
#score_vector_flex = ['fa_atr','fa_elec','fa_rep','fa_sol','hbond_bb_sc','hbond_lr_bb','hbond_sc']
#score_vector_flex = ['fa_atr','fa_elec','fa_intra_rep','fa_rep','fa_sol'] 

score_vector_flex_groups = {'fg1': score_vector_flex}

original = True

res_path = 'Results/'    
#mode 0: just flex
#mode 1: just lisa
#mode 2: flex and lisa

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def ModelEvaluation(model,XtR,YtR,XtS,YtS):
    startT = time.time()
    model.fit(XtR, YtR)
    #res = model.score(XtS, YtS)
    predictions = model.predict(XtS)
    endT = time.time()
    res = r2_score(YtS, predictions)
    return res,endT-startT

def PreprocessData(X_mt, X_wt, features):
	# Performin min-max scaling each continuous feature column to the range [0, 1]
    # The min and max are extracted from MT!
	cnts = MinMaxScaler()
	X_mt_scaled = cnts.fit_transform(X_mt[features])
	X_wt_scaled = cnts.transform(X_wt[features])
	return pd.concat([X_mt[inden_vector_skmp],pd.DataFrame(data=X_mt_scaled, columns=features)],axis=1), \
            pd.concat([X_wt[inden_vector_skmp_wt],pd.DataFrame(data=X_wt_scaled, columns=features)],axis=1)


def FetchData(name, feat_to_drop, do_preprocess):
    source = "Backrub"
    output_files = os.path.join(res_path, source + '_' + name)
    results_flex_wt = pd.read_csv(output_files + '_results_flex_wt.csv', sep=';')
    results_flex_mt = pd.read_csv(output_files + '_results_flex_mt.csv', sep=';')
    
    results_flex_total_wt = pd.read_csv(output_files + '_results_flex_total_wt.csv', sep=';')
    results_flex_total_mt = pd.read_csv(output_files + '_results_flex_total_mt.csv', sep=';')
    score_vector_f = score_vector_flex.copy()
    score_vector_f_groups = score_vector_flex_groups.copy()
    
    if not original:
        results_lisa_total_wt = pd.read_csv(output_files + '_results_lisa_total_wt.csv', sep=';')
        results_lisa_total_mt = pd.read_csv(output_files + '_results_lisa_total_mt.csv', sep=';')
        score_vector_l = score_vector_lisa.copy()
    else:
        results_lisa_total_wt = pd.read_csv(output_files + '_results_lisa_total_wt_original.csv', sep=';')
        results_lisa_total_mt = pd.read_csv(output_files + '_results_lisa_total_mt_original.csv', sep=';')
        score_vector_l = score_vector_lisa_original.copy()
    score_vector_l_groups = score_vector_lisa_original_groups.copy()
    
    for feat in feat_to_drop:
        if feat in score_vector_l: score_vector_l.remove(feat)
        if feat in score_vector_f: score_vector_f.remove(feat)
        
        for gr in score_vector_l_groups:
            if feat in score_vector_l_groups[gr]: score_vector_l_groups[gr].remove(feat)
        for gr in score_vector_f_groups:
            if feat in score_vector_f_groups[gr]: score_vector_f_groups[gr].remove(feat)
    
    if do_preprocess:
        results_flex_mt, results_flex_wt = PreprocessData(results_flex_mt, results_flex_wt, score_vector_f)
        results_flex_total_mt, results_flex_total_wt = PreprocessData(results_flex_total_mt, results_flex_total_wt, score_vector_f)
        results_lisa_total_mt, results_lisa_total_wt = PreprocessData(results_lisa_total_mt, results_lisa_total_wt, score_vector_l)
    
    return results_flex_wt, results_flex_mt, \
           results_flex_total_wt, results_flex_total_mt, \
           results_lisa_total_wt, results_lisa_total_mt, \
           score_vector_l, score_vector_f, \
           score_vector_l_groups, score_vector_f_groups

def SerializeData(X_df, features):
    s_data = np.array([])
    for feat in features:
        s_data = np.concatenate((s_data,X_df[feat].to_numpy()))
    return s_data
           
    
def FeatureVectorCreator(name, mode, feat_to_drop, do_preprocess, blindtest):
    
    results_flex_wt, results_flex_mt, \
    results_flex_total_wt, results_flex_total_mt, \
    results_lisa_total_wt, results_lisa_total_mt, \
    score_vector_l, score_vector_f, \
    score_vector_l_groups, score_vector_f_groups = FetchData(name, feat_to_drop, do_preprocess)
    
    dataX_wt = []
    dataX_mt = []
    dataX_ddg = []
    
    dataX_total_wt = []
    dataX_total_mt = []
    dataX_total_ddg = []
    
    dataX_total_group_wt = []
    dataX_total_group_mt = []
    dataX_total_group_ddg = []

    dataY_wt = []
    dataY_mt = []
    dataY_ddg = []
    
    
    blindtest_flex_mt = pd.DataFrame(columns = inden_vector_skmp + score_vector_f)
    blindtest_flex_wt = pd.DataFrame(columns = inden_vector_skmp_wt + score_vector_f)
    blindtest_lisa_mt = pd.DataFrame(columns = inden_vector_skmp + score_vector_lisa)
    blindtest_lisa_wt = pd.DataFrame(columns = inden_vector_skmp_wt + score_vector_lisa)
        
    wt_already_processed = [] #To avoid redundancy in the WT samples!
    
    complex_data_blocks = {}
    
    samples_dataframe = results_flex_mt[inden_vector_skmp]
    for index, row in samples_dataframe.iterrows():
        protein_complex = str(row['#Pdb'])
        mutate_complex = str(row['Mutation(s)_cleaned'])
        interaction_region = str(row['iMutation_Location(s)'])
        complex_type = str(row['Hold_out_type'])
        experimental_method = str(row['Method'])
        b_affine_wt = float(row['Affinity_wt'])
        b_affine_mt = float(row['Affinity_mut'])
        
        #XXX FIX IT!!! Find out what is wrong with these structures. It influences the ddg in npy files because they have data for mutations but not for wild-type!!
        if(protein_complex == '4N8V_G_ABC' or 
           protein_complex == '1SBB_A_B' or 
           protein_complex == '2KSO_A_B' or 
           protein_complex == '3UII_A_P' or 
           protein_complex == '5K39_A_B' or 
           protein_complex == '5M2O_A_B'): 
            continue
        
        if blindtest and random.randint(0,100) < 30: #30 percent of samples for the blindtest
            flex_features_wt = results_flex_wt[results_flex_wt['#Pdb'] == protein_complex]
            #flex_features_wt = flex_features_wt[flex_features_wt['Mutation(s)_cleaned'] == mutate_complex]
            
            flex_features_mt = results_flex_mt[results_flex_mt['#Pdb'] == protein_complex]
            flex_features_mt = flex_features_mt[flex_features_mt['Mutation(s)_cleaned'] == mutate_complex]
            
            lisa_features_wt = results_lisa_total_wt[results_lisa_total_wt['#Pdb'] == protein_complex]
            #lisa_features_wt = lisa_features_wt[lisa_features_wt['Mutation(s)_cleaned'] == mutate_complex]
            
            lisa_features_mt = results_lisa_total_mt[results_lisa_total_mt['#Pdb'] == protein_complex]
            lisa_features_mt = lisa_features_mt[lisa_features_mt['Mutation(s)_cleaned'] == mutate_complex]
            
            blindtest_flex_wt = pd.concat([blindtest_flex_wt,flex_features_wt])
            blindtest_flex_mt = pd.concat([blindtest_flex_mt,flex_features_mt])
            blindtest_lisa_wt = pd.concat([blindtest_lisa_wt,lisa_features_wt])
            blindtest_lisa_mt = pd.concat([blindtest_lisa_mt,lisa_features_mt])
            
            continue
        
        process_wt = False
        if protein_complex not in wt_already_processed:
            process_wt = True
            wt_already_processed.append(protein_complex)
        
        
        if process_wt:
            dataY_wt.append(b_affine_wt)
        dataY_mt.append(b_affine_mt)
        dataY_ddg.append(b_affine_mt-b_affine_wt)

        flex_features_wt = results_flex_wt[results_flex_wt['#Pdb'] == protein_complex]
        flex_features_wt = flex_features_wt[score_vector_f]
        flex_features_total_wt = results_flex_total_wt[results_flex_total_wt['#Pdb'] == protein_complex]
        flex_features_total_wt = flex_features_total_wt[score_vector_f]
        _nstruct = len(flex_features_total_wt)

        
        dataX_total_group_wt_tmp = []
        #for gr in score_vector_f_groups:
        #    dataX_wt_total_tmp_groups[gr] = np.array([]).reshape(_nstruct*len(score_vector_f_groups[gr]),0)
        
        if mode in [0,2]:
            dataX_wt_tmp = flex_features_wt.values[0]
            dataX_total_wt_tmp = flex_features_total_wt.to_numpy()
            
            ###
            for gr_feat in score_vector_f_groups.values():
                dataX_total_group_wt_tmp.append(SerializeData(flex_features_total_wt, gr_feat))
            ####    
        else:
            dataX_wt_tmp = np.array([])
            dataX_total_wt_tmp = np.array([]).reshape(_nstruct,0)



        flex_features_mt = results_flex_mt[results_flex_mt['#Pdb'] == protein_complex]
        flex_features_mt = flex_features_mt[flex_features_mt['Mutation(s)_cleaned'] == mutate_complex]
        flex_features_mt = flex_features_mt[score_vector_f]
        flex_features_total_mt = results_flex_total_mt[results_flex_total_mt['#Pdb'] == protein_complex]
        flex_features_total_mt = flex_features_total_mt[flex_features_total_mt['Mutation(s)_cleaned'] == mutate_complex]
        flex_features_total_mt = flex_features_total_mt[score_vector_f]
        
        dataX_total_group_mt_tmp = []
        if mode in [0,2]:
            dataX_mt_tmp = flex_features_mt.values[0]
            dataX_total_mt_tmp = flex_features_total_mt.to_numpy()
            ###
            for gr_feat in score_vector_f_groups.values():
                dataX_total_group_mt_tmp.append(SerializeData(flex_features_total_mt, gr_feat))
            ####  
        else:
            dataX_mt_tmp = np.array([])
            dataX_total_mt_tmp = np.array([]).reshape(_nstruct,0)



        lisa_features_wt = results_lisa_total_wt[results_lisa_total_wt['#Pdb'] == protein_complex]
        lisa_features_total_wt = lisa_features_wt[score_vector_l]
        lisa_features_wt = lisa_features_total_wt.mean()
        
        if mode in [1,2]:
            dataX_wt_tmp = np.concatenate((dataX_wt_tmp,lisa_features_wt.values))
            dataX_total_wt_tmp = np.concatenate((dataX_total_wt_tmp,lisa_features_total_wt.to_numpy()),axis=1)
            
            ###
            for gr_feat in score_vector_l_groups.values():
                dataX_total_group_wt_tmp.append(SerializeData(lisa_features_total_wt, gr_feat))
            #### 
            
            

        lisa_features_mt = results_lisa_total_mt[results_lisa_total_mt['#Pdb'] == protein_complex]
        lisa_features_mt = lisa_features_mt[lisa_features_mt['Mutation(s)_cleaned'] == mutate_complex]
        lisa_features_total_mt = lisa_features_mt[score_vector_l]
        lisa_features_mt = lisa_features_total_mt.mean()
        
        if mode in [1,2]:
            dataX_mt_tmp = np.concatenate((dataX_mt_tmp,lisa_features_mt.values))
            dataX_total_mt_tmp = np.concatenate((dataX_total_mt_tmp,lisa_features_total_mt.to_numpy()),axis=1)
            ###
            for gr_feat in score_vector_l_groups.values():
                dataX_total_group_mt_tmp.append(SerializeData(lisa_features_total_mt, gr_feat))
            ####
            
            
        if process_wt:
            dataX_wt.append(dataX_wt_tmp)
            dataX_total_wt.append(dataX_total_wt_tmp)
            dataX_total_group_wt.append(dataX_total_group_wt_tmp)
        
        dataX_mt.append(dataX_mt_tmp)
        dataX_total_mt.append(dataX_total_mt_tmp)
        dataX_total_group_mt.append(dataX_total_group_mt_tmp)
        
        dataX_ddg.append(dataX_mt_tmp - dataX_wt_tmp)
        dataX_ddg_total_tmp = dataX_total_mt_tmp - dataX_total_wt_tmp
        dataX_total_ddg.append(dataX_ddg_total_tmp)
        
        
        dataX_total_group_ddg.append([dataX_total_group_mt_tmp[gr] - dataX_total_group_wt_tmp[gr] 
                                      for gr in range(len(dataX_total_group_mt_tmp))])
        
        if protein_complex in complex_data_blocks:
            complex_data_blocks[protein_complex].append(dataX_ddg_total_tmp)
        else:
            complex_data_blocks[protein_complex] = [dataX_ddg_total_tmp]
    
    FeatNames_groups = {}
    if mode == 0:
        FeatNames = score_vector_f
        for gr, gr_feat  in score_vector_f_groups.items():
            FeatNames_groups[gr] = len(gr_feat)
            
    elif mode == 1:
        FeatNames = score_vector_l
        for gr, gr_feat  in score_vector_l_groups.items():
            FeatNames_groups[gr] = len(gr_feat)
            
    elif mode == 2:
        FeatNames = score_vector_f + score_vector_l
        for gr, gr_feat  in score_vector_f_groups.items():
            FeatNames_groups[gr] = len(gr_feat)
        for gr, gr_feat  in score_vector_l_groups.items():
            FeatNames_groups[gr] = len(gr_feat)
        
    #FeatNames_ddg = FeatNames
    
    return np.array(dataX_wt),np.array(dataX_mt),np.array(dataX_ddg), \
           np.array(dataX_total_wt),np.array(dataX_total_mt),np.array(dataX_total_ddg), \
           dataX_total_group_wt,dataX_total_group_mt,dataX_total_group_ddg, \
           np.array(dataY_wt),np.array(dataY_mt),np.array(dataY_ddg), \
           FeatNames, FeatNames_groups, \
           complex_data_blocks, \
           blindtest_flex_wt,blindtest_flex_mt,blindtest_lisa_wt,blindtest_lisa_mt

def FeatureVectorCreatorXRay(name):
    source = "XRay"
    output_files = res_path + source + '_' + name
    
    if not original:
        features_lisa_xray_wt = pd.read_csv(output_files + '_results_lisa_wt.csv', sep=';')
        score_vector = score_vector_lisa
    else:
        features_lisa_xray_wt = pd.read_csv(output_files + '_results_lisa_wt_original.csv', sep=';')
        score_vector = score_vector_lisa_original
    
    dataX_xr = []
    dataY_xr = []
    for index, row in features_lisa_xray_wt.iterrows():
        protein_complex = str(row['#Pdb'])
        mutate_complex = str(row['Mutation(s)_cleaned'])
        interaction_region = str(row['iMutation_Location(s)'])
        complex_type = str(row['Hold_out_type'])
        experimental_method = str(row['Method'])
        b_affine_wt = float(row['Affinity_wt'])
        b_affine_mt = float(row['Affinity_mut'])
        
        dataY_xr.append(b_affine_wt)
        
        lisa_features_xr = features_lisa_xray_wt[features_lisa_xray_wt['#Pdb'] == protein_complex]
        lisa_features_xr = lisa_features_xr[score_vector]                                                               
        dataX_xr.append(lisa_features_xr.values[0])
        
    FeatNames = score_vector
    return np.array(dataX_xr), np.array(dataY_xr), FeatNames




"""
Pinciple Component Analysis

def PincipleComponentAnalysis(nComp=2):
    print('\n\n')
    print("##############################################")
    print("####### Pinciple Component Analysis ##########")
    print("##############################################")
    pca = PCA(n_components=nComp)
    pca.fit(dataX)
    dataX_pca = pca.transform(dataX)
    if(nComp == 2):
        plt.figure()
        plt.scatter(dataX_pca[:, 0], dataX_pca[:, 1], marker='o', c=dataY, s=25, edgecolor='k')
    elif(nComp ==3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dataX_pca[:, 0], dataX_pca[:, 1], dataX_pca[:, 2], marker='o', c=dataY, edgecolor='k')
    #plt.show()
    plt.savefig('pca_analysis')
"""

"""
Cross-Validation
"""
def CrossValidation(dataX, dataY, mode):
    print('\n\n')
    print("##############################################")
    print("############# Cross-Validation ###############")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
    elif mode == 3:
        mode = "LisaXRay"
          
          
    clf_xgboost_cv = xgb.XGBRegressor(max_depth=mDepth, n_estimators=nTrees, learning_rate=lRate, booster='gbtree')
    clf_skgb_cv = GradientBoostingRegressor(n_estimators=nTrees, learning_rate=lRate, max_depth=mDepth, random_state=0)
    clf_skrf_cv = RandomForestRegressor(n_estimators=nTrees, max_depth=mDepth, random_state=0)
    clf_skab_cv = AdaBoostRegressor(n_estimators=nTrees, learning_rate=lRate, random_state=0)    
    
    #dataX = np.array(dataX)
    #dataY = np.array(dataY)
    
    #skf = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    skf = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_acc_skgb = []
    cv_acc_skrf = []
    cv_acc_skab = []
    cv_acc_xgboost = []
    totalDur_skgb = 0
    totalDur_skrf = 0
    totalDur_skab = 0
    totalDur_xgb = 0
    for train_index, test_index in skf.split(dataX, dataY):
        
        X_train, X_test = dataX[train_index], dataX[test_index]
        y_train, y_test = dataY[train_index], dataY[test_index]
        
        res_skgb,dur_skgb = ModelEvaluation(clf_skgb_cv,X_train,y_train,X_test,y_test)
        #print("Scikit Gradient Boosting Score: %0.2f Time: %f" % (res_skgb,dur_skgb))
        cv_acc_skgb.append(res_skgb)
        totalDur_skgb += dur_skgb
        
        res_xgb, dur_xgb = ModelEvaluation(clf_xgboost_cv,X_train,y_train,X_test,y_test)
        #print("XGBoost Score: %0.2f Time: %f" % (res_xgb, dur_xgb))
        cv_acc_xgboost.append(res_xgb)
        totalDur_xgb += dur_xgb
        
        res_skrf,dur_skrf = ModelEvaluation(clf_skrf_cv,X_train,y_train,X_test,y_test)
        #print("Scikit Random Forest Score: %0.2f Time: %f" % (res_skrf,dur_skrf))
        cv_acc_skrf.append(res_skrf)
        totalDur_skrf += dur_skrf
        
        res_skab,dur_skab = ModelEvaluation(clf_skab_cv,X_train,y_train,X_test,y_test)
        #print("Scikit AdaBoost Score: %0.2f Time: %f" % (res_skab,dur_skab))
        cv_acc_skab.append(res_skab)
        totalDur_skab += dur_skab        
        #print('\n')
    
    cv_acc_skgb = np.array(cv_acc_skgb)
    cv_acc_skrf = np.array(cv_acc_skrf)
    cv_acc_skab = np.array(cv_acc_skab)
    cv_acc_xgboost = np.array(cv_acc_xgboost)
    print("\nSciKit Gradient Boosting CV Accuracy: %0.2f (+/- %0.2f), total time: %f" % (cv_acc_skgb.mean(), cv_acc_skgb.std() * 2, totalDur_skgb))
    print("XGBoost CV Accuracy: %0.2f (+/- %0.2f), total time: %f" % (cv_acc_xgboost.mean(), cv_acc_xgboost.std() * 2, totalDur_xgb))
    print("SciKit Random Forest CV Accuracy: %0.2f (+/- %0.2f), total time: %f" % (cv_acc_skrf.mean(), cv_acc_skrf.std() * 2, totalDur_skrf))
    print("SciKit Adaboost CV Accuracy: %0.2f (+/- %0.2f), total time: %f" % (cv_acc_skab.mean(), cv_acc_skab.std() * 2, totalDur_skab))


"""
Performances
"""
def Performance(X_train, X_test, y_train, y_test, FeatNames, source, mode):
    print('\n\n')
    print("##############################################")
    print("############### Performances #################")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
    elif mode == 3:
        mode = "LisaXRay"
          
        
    res_skgb = np.zeros((T))
    res_skrf = np.zeros((T))
    res_skab = np.zeros((T))
    res_xgb = np.zeros((T))

    dur_skgb = np.zeros((T))
    dur_skrf = np.zeros((T))
    dur_skab = np.zeros((T))
    dur_xgb = np.zeros((T))    
    
    totalDur_skgb = 0
    totalDur_skrf = 0
    totalDur_skab = 0
    totalDur_xgb = 0
    for t in range(1,T):
        #print("Iteration t=",t)
        clf_skgb_pr = GradientBoostingRegressor(n_estimators=t, learning_rate=lRate, max_depth=mDepth, random_state=0)
        res_skgb[t], dur_skgb[t] = ModelEvaluation(clf_skgb_pr,X_train,y_train,X_test,y_test)
        totalDur_skgb += dur_skgb[t]
        
        clf_xgb_pr = xgb.XGBRegressor(max_depth=mDepth, n_estimators=t, learning_rate=lRate, booster='gbtree')
        res_xgb[t], dur_xgb[t] = ModelEvaluation(clf_xgb_pr,X_train,y_train,X_test,y_test)
        totalDur_xgb += dur_xgb[t]
        
        clf_skrf_pr = RandomForestRegressor(n_estimators=t, max_depth=mDepth, random_state=0)
        res_skrf[t], dur_skrf[t] = ModelEvaluation(clf_skrf_pr,X_train,y_train,X_test,y_test)
        totalDur_skrf += dur_skrf[t]   
        
        clf_skab_pr = AdaBoostRegressor(n_estimators=t, learning_rate=lRate, random_state=0)
        res_skab[t], dur_skab[t] = ModelEvaluation(clf_skab_pr,X_train,y_train,X_test,y_test)
        totalDur_skab += dur_skab[t]
        
    print('\n')
    print('Time_scikit gradient boosting: '+str(totalDur_skgb)) 
    print('Time_xgb: '+str(totalDur_xgb))
    print('Time_scikit random forest: '+str(totalDur_skrf))
    print('Time_scikit adaboost: '+str(totalDur_skab))
    
    plt.figure()
    plt.plot(res_skgb, label="SciKit GB", linestyle='--')
    plt.plot(res_skrf, label="SciKit RF", linestyle='--')
    plt.plot(res_skab, label="SciKit AB", linestyle='--')
    plt.plot(res_xgb, label="XGBoost")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy score')
    plt.grid()
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(res_path + "performance_accuracy" + source + "_" + mode)
    
    plt.figure()
    plt.plot(dur_skgb, label="SciKit GB", linestyle='--')
    plt.plot(dur_skrf, label="SciKit RF", linestyle='--')
    plt.plot(dur_skab, label="SciKit AB", linestyle='--')
    plt.plot(dur_xgb, label="XGBoost")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Number of trees')
    plt.ylabel('Experiment time (s)')
    plt.grid()
    #plt.show()
    plt.savefig(res_path + "performance_time_" + source + "_" + mode)


def f_importances(coef, names, source, mode):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.savefig(res_path+'__features_selection_'+source+"_"+mode)


"""
Feature-Selection
"""
def FeatureSelection(X_train, X_test, y_train, y_test, FeatNames, source, mode):
    print('\n\n')
    print("##############################################")
    print("############# Feature-Selection ##############")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
    elif mode == 3:
        mode = "LisaXRay"
    
    #Logistic regression penalized by the L1 penalty term 
    print('\n')
    print('Logistic regression (penalized by the L1)')
    clf_lasso = Lasso(alpha=0.01)
    res_lasso, dur_lasso = ModelEvaluation(clf_lasso,X_train,y_train,X_test,y_test)
    print("Scikit Lasso Score: %0.2f Time: %f" % (res_lasso,dur_lasso))
    clf_lasso_importance = np.abs(clf_lasso.coef_) / sum(np.abs(clf_lasso.coef_))
    #clf_lasso.fit(dataX, dataY)
    #print("Selected features: ", len([x for x in clf_lasso.coef_ if x != 0]))
    #predicted = cross_val_predict(clf_lasso, dataX, dataY, cv=10) 
    #print("Score: ", accuracy_score(dataY, np.sign(np.array(predicted))))
    
    
    #A support vector regression penalized by the L1 penalty term
    warnings.simplefilter("ignore")
    print('\n')
    print('SVR (penalized by the L1)')
    clf_svr = SVR(kernel="linear")
    res_svr, dur_svr = ModelEvaluation(clf_svr,X_train,y_train,X_test,y_test)
    print("Scikit SVR Score: %0.2f Time: %f" % (res_svr,dur_svr))
    clf_svr_importance = np.abs(clf_svr.coef_[0]) / sum(np.abs(clf_svr.coef_[0]))
    #print("Selected features: ", len([x for x in clf_svr.coef_[0] if x != 0]))
    #predicted = cross_val_predict(clf_svr, dataX, dataY, cv=10) 
    #print("Score: ", accuracy_score(dataY, np.sign(np.array(predicted))))
    #print("Score: ", clf_svr.score(dataX, dataY))
    #f_importances(clf_svr.coef_[0], FeatNames, source)
    
    #selector = RFECV(clf_svr, cv=5, step=1)
    #selector = selector.fit(dataX, dataY)
    #print(selector.support_)
    #print(selector.ranking_)
    
       
    #Explore the Elastic Net which is a compromise between the L1 and L2 penalty terms.
    print('\n')
    print('Elastic Net (compromise between the L1 and L2)')
    clf_elastic = ElasticNet(alpha=0.6, l1_ratio=0.2)
    res_elastic, dur_elastic = ModelEvaluation(clf_elastic,X_train,y_train,X_test,y_test)
    print("Scikit Elastic Score: %0.2f Time: %f" % (res_elastic,dur_elastic))
    clf_elastic_importance = np.abs(clf_elastic.coef_) / sum(np.abs(clf_elastic.coef_))
    #clf_elastic.fit(dataX, dataY)
    #print("Selected features: ", len([x for x in clf_elastic.coef_ if x != 0]))
    #predicted = cross_val_predict(clf_elastic, dataX, dataY, cv=10) 
    #print("Score: ", accuracy_score(dataY, np.sign(np.array(predicted))))
    
    
    #SciKit Gradient Boosting
    print('\n')
    clf_skgb = GradientBoostingRegressor(n_estimators=nTrees, learning_rate=lRate, max_depth=mDepth, random_state=0)
    res_skgb, dur_skgb = ModelEvaluation(clf_skgb,X_train,y_train,X_test,y_test)
    save_obj(clf_skgb, folder_GBM+'/model_fold'+str(0))
    print("Scikit Gradient Boosting Score: %0.2f Time: %f" % (res_skgb,dur_skgb))
    
    #SciKit Random Forest
    print('\n')
    clf_skrf = RandomForestRegressor(n_estimators=nTrees, max_depth=mDepth, random_state=0)
    res_skrf, dur_skrf = ModelEvaluation(clf_skrf,X_train,y_train,X_test,y_test)  
    print("Scikit Random Forest Score: %0.2f Time: %f" % (res_skrf,dur_skrf))
    
    #SciKit AdaBoost
    print('\n')
    clf_skab = AdaBoostRegressor(n_estimators=nTrees, learning_rate=lRate, random_state=0)
    res_skab, dur_skab = ModelEvaluation(clf_skab,X_train,y_train,X_test,y_test)  
    print("Scikit AdaBoost Score: %0.2f Time: %f" % (res_skab,dur_skab))

    #XGBoost
    print('\n')
    clf_xgb = xgb.XGBRegressor(max_depth=mDepth, n_estimators=nTrees, learning_rate=lRate, booster='gbtree')
    res_xgb, dur_xgb = ModelEvaluation(clf_xgb,X_train,y_train,X_test,y_test)
    save_obj(clf_xgb, folder_XGB+'/model_fold'+str(0))
    print("XGBoost Score: %0.2f Time: %f" % (res_xgb, dur_xgb))

    
    plt.figure(figsize=(15,8))
    
    nFeats = len(X_train[0])
    xaxis = np.arange(nFeats)
    width = 0.1    
    margin = 0.2
    #width = (1.-2.*margin)/nFeats
    
    plt.bar(xaxis+margin+0*width, clf_skgb.feature_importances_, width, label = "scikit GB")
    plt.bar(xaxis+margin+1*width, clf_xgb.feature_importances_, width, label = "XGBoost")
    plt.bar(xaxis+margin+2*width, clf_skrf.feature_importances_, width, label = "RandomForest")
    plt.bar(xaxis+margin+3*width, clf_skab.feature_importances_, width, label = "AdaBoost")
    plt.bar(xaxis+margin+4*width, clf_lasso_importance, width, label = "Lasso")
    plt.bar(xaxis+margin+5*width, clf_elastic_importance, width, label = "ElasticNet")
    plt.bar(xaxis+margin+6*width, clf_svr_importance, width, label = "SVR")
    
    plt.xticks(xaxis+0.5, FeatNames, rotation=45)
    plt.title('Feature Selection')
    plt.legend()
    #plt.show()
    plt.savefig(res_path+'features_selection_'+source+"_"+mode)
    
    xgb.plot_importance(clf_xgb)
    plt.title('xgb')
    #plt.show()
    plt.savefig(res_path+'_features_xgb_'+source+"_"+mode)
    
    #This part is not so necessary!
    """
    print('\n\n')
    print("##############################################")
    print("################# Save Trees #################")
    print("##############################################")
    for i in range(nTrees):
        sub_tree = clf_skgb.estimators_[i, 0]
        dot_data = tree.export_graphviz(
            sub_tree,
            out_file=None, filled=True,
            rounded=True,  
            special_characters=True,
            proportion=True)
        graph = graphviz.Source(dot_data) 
        graph.render(folder_GBM+'/t'+str(i)+'.gv', view=False)
    
        graph = xgb.to_graphviz(clf_xgb, num_trees=i)
        graph.render(folder_XGB+'/t'+str(i)+'.gv', view=False)
    """    
    
#PincipleComponentAnalysis(2) 
    


def DoCorrelationPlot(b_affinities_expr, b_affinities_pred, output_files, ddG, mutated, source, name):
    #Do PLOT
    print(b_affinities_expr)
    print(b_affinities_pred)
    
    v_type = "$\Delta$G"
    l = -20
    u = 0
    if ddG == True:
        l = -10
        u = 15
        v_type = "$\Delta\Delta$G"

    x = []
    y = []
    scatter_x = dict()
    scatter_y = dict()
    pmarker = {'Pr/PI': 'o', 'AB/AG': 'd', 'TCR/pMHC': '*'}
    pcolor1 = {'ITC': 'y', 'SPR':'c', 'SP': 'b', 'FL': 'r'}
    pcolor2 = {'INT': 'y', 'SUR':'c', 'COR': 'b', 'SUP': 'r', 'RIM': 'g', 'MLT': 'm'}
    plt.figure(figsize=(10,7))
    wt_already_processed = [] #To avoid redundancy in the WT samples!
    for i in range(len(b_affinities_expr)):
        
        expr_value = float(b_affinities_expr[i][0]) - 0.0
        region     = b_affinities_expr[i][1]
        pctype     = b_affinities_expr[i][2]
        method     = b_affinities_expr[i][3]
        pccode     = b_affinities_expr[i][4]
        mtcode     = b_affinities_expr[i][5]
        pred_value = float(b_affinities_pred[i][0]) - 0.0
        
        if not mutated:
            if pccode in wt_already_processed:
                continue
            else:
                wt_already_processed.append(pccode)
        
        #Some exceptions in the SKEMPI database
        pctype = 'Unknown' if pctype == 'nan' else 'AB/AG' if pctype == 'AB/AG,Pr/PI' else pctype 
        method = 'Unknown' if method == 'nan' else method
        region = 'Unknown' if region == 'nan' else 'MLT' if len(region.split(',')) > 1 else region
        
        
        #Filtering for different representation of results!
        """
        #Based on mutations between groups of amino acids
        if mutated:
            if region == 'MLT':
                continue            
            original_aa = mtcode[0]
            mutated_ch = mtcode[1]
            position_aa = mtcode[2:-1]
            mutated_aa = mtcode[-1]
            if original_aa not in aminoacid_classes['nonpolar']:
                continue
            if mutated_aa not in aminoacid_classes['polar']:
                continue
            
        #Based on experimental method
        #if not method == 'ITC':
        #    continue
        """
        
        scatter_x.setdefault((region, pctype, method), list()).append(expr_value)
        scatter_y.setdefault((region, pctype, method), list()).append(pred_value)
        x.append(expr_value)
        y.append(pred_value)
    
    #Labels are always colors (for wildtype label is experimental method and for mutant label is region of mutation)
    #Markers are always complex types
    labels = []
    for config in scatter_x:        
        lbl = config[2]
        #clr = 'b' if config[1] == 'Pr/PI' else 'r' if config[1] == 'AB/AG' else 'g' if config[1] == 'TCR/pMHC' else 'black'
        clr = pcolor1[lbl] if pcolor1.get(lbl) != None else 'k'
        if mutated:
            #clr = pcolor[config[0]] if pcolor.get(config[0]) != None else 'k'
            #lbl = config[0]
            #lbl = config[0][:3] #Because sometimes we have several mutations on different regions such as this case: SUP,SUP,COR,COR (here we takes the first mutation!!!)
            lbl = config[0]
            clr = pcolor2[lbl] if pcolor2.get(lbl) != None else 'k'
        
        if lbl in labels:
            lbl = ""
        else:
            labels.append(lbl)
        
        mrk = pmarker[config[1]] if pmarker.get(config[1]) != None else 'v'
        plt.scatter(scatter_x[config], scatter_y[config], color = clr, marker = mrk, label = lbl)
    
    x = np.array(x)
    y = np.array(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)    
    line = slope*np.array(x)+intercept
    plt.plot(x, line)
    corr_pearson = stats.pearsonr(x,y)
    corr_spearman = stats.spearmanr(x,y)
    
    plt.legend()
    plt.title(v_type + ' Prediction vs Experiment ' + '(' + source + ')_' + name)
    plt.text(-7, 12, 'Pearson: R=' + str(np.round(corr_pearson[0],2)) + ', p=' + '{:.1e}'.format(corr_pearson[1]) + 
                      '\nSpearman: R=' + str(np.round(corr_spearman[0],2)) + ', p=' + '{:.1e}'.format(corr_spearman[1]))
    plt.xlabel('Experimental ' + v_type + ' (kcal/mol)')
    plt.ylabel('Predicted ' + v_type + ' (kcal/mol)')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.grid()
    plt.savefig(output_files)  



#XXX needs updates!!
def BestModelEvaluation(name):
    source='BestModel'
    output_files = res_path + source + '_' + name
    b_affinities_pred_ddg = []
    b_affinities_expr_ddg = []
    
    if original:
        score_vector_l = score_vector_lisa_original
    else:
        score_vector_l = score_vector_lisa
    
    #Building model
    dataX_wt,dataX_mt,dataX_ddg,dataY_wt,dataY_mt,dataY_ddg,FeatNames,FeatNames_ddg,btest_flex_wt,btest_flex_mt,btest_lisa_wt,btest_lisa_mt = FeatureVectorCreator('ALL', 2, True)
    X_gl_train = np.concatenate((dataX_wt,dataX_mt))
    Y_gl_train = np.concatenate((dataY_wt,dataY_mt))
    
    clf_xgboost = xgb.XGBRegressor(max_depth=mDepth, n_estimators=nTrees, learning_rate=lRate, booster='gbtree')
    clf_xgboost.fit(X_gl_train, Y_gl_train)
    
    
    for index, row in btest_flex_mt.iterrows():
        protein_complex = str(row['#Pdb'])
        mutate_complex = str(row['Mutation(s)_cleaned'])
        interaction_region = str(row['iMutation_Location(s)'])
        complex_type = str(row['Hold_out_type'])
        experimental_method = str(row['Method'])
        b_affine_wt = float(row['Affinity_wt'])
        b_affine_mt = float(row['Affinity_mut'])
        
        flex_features_wt = btest_flex_wt[btest_flex_wt['#Pdb'] == protein_complex]
        flex_features_wt = flex_features_wt[flex_features_wt['Mutation(s)_cleaned'] == mutate_complex]
        flex_features_wt = flex_features_wt[score_vector_flex]
        featureX_wt = flex_features_wt.values[0]

        flex_features_mt = btest_flex_mt[btest_flex_mt['#Pdb'] == protein_complex]
        flex_features_mt = flex_features_mt[flex_features_mt['Mutation(s)_cleaned'] == mutate_complex]
        flex_features_mt = flex_features_mt[score_vector_flex]
        featureX_mt = flex_features_mt.values[0]

        lisa_features_wt = btest_lisa_wt[btest_lisa_wt['#Pdb'] == protein_complex]
        lisa_features_wt = lisa_features_wt[lisa_features_wt['Mutation(s)_cleaned'] == mutate_complex]
        lisa_features_wt = lisa_features_wt[score_vector_l].mean()
        featureX_wt = np.concatenate((featureX_wt,lisa_features_wt.values))

        lisa_features_mt = btest_lisa_mt[btest_lisa_mt['#Pdb'] == protein_complex]
        lisa_features_mt = lisa_features_mt[lisa_features_mt['Mutation(s)_cleaned'] == mutate_complex]
        lisa_features_mt = lisa_features_mt[score_vector_l].mean()
        featureX_mt = np.concatenate((featureX_mt,lisa_features_mt.values))
        
        
        featureX_wt = np.array([featureX_wt])
        featureX_mt = np.array([featureX_mt])
        b_affine_pred_wt = clf_xgboost.predict(featureX_wt)[0]
        b_affine_pred_mt = clf_xgboost.predict(featureX_mt)[0]
        
        b_affinities_pred_ddg.append((b_affine_pred_mt-b_affine_pred_wt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
        b_affinities_expr_ddg.append((b_affine_mt-b_affine_wt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
    
    DoCorrelationPlot(b_affinities_expr_ddg, b_affinities_pred_ddg, output_files+'_ddg', True, True, source + 'DlDlG', name)








#New functions
########################################################""
    
# find the largest target value in the training set and use it to
# scale target values to the range [0, 1] (will lead to better
# training and convergence)

#maxVal = trainAttrX["target"].max()
#trainY = trainAttrX["target"] / maxVal
#testY = testAttrX["target"] / maxVal

#Then use PreprocessData to process attributes and features!!
# process the attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together    
    
    
# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
    
#"mean_absolute_percentage_error" : maybe for normalized target we should use this!!
    
"""    
def PreprocessData(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["bedrooms", "bathrooms", "area"]
	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])
	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoding, all output features are now in the range [0, 1])
	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])
	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])
	# return the concatenated training and testing data
	return (trainX, testX)
"""


def CreateMLP(dim, regress=False):
	# Define a MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# Check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# Return our model
	return model
########################################################"""


def DoKerasMagic(X_total_train, X_total_test, y_total_train, y_total_test, FeatNames, source, mode):
    print('\n\n')
    print("##############################################")
    print("############# Keras-Training ##############")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
    
    print(len(X_total_train))
    print(len(y_total_train))
    
    in_dim = 30
    branches_in = []
    branches_out = []
    
    # Each branch operates on each feature (each input vector belongs to a feature)
    for feat in FeatNames:
        inputL = Input(shape=(in_dim,))
        x = None
        x = Dense(in_dim, activation="relu")(inputL)
        #x = Dense(in_dim, activation="relu")(x)
        x = Model(inputs=inputL, outputs=x)
        branches_in.append(x.input)
        branches_out.append(x.output)
    
    # Combine the output of all branches    
    combined = concatenate(branches_out)
    
    # Apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(100, activation="relu")(combined)
    #z = Dense(100, activation="relu")(z)
    z = Dense(40, activation="relu")(z)
    z = Dense(8, activation="relu")(z)
    z = Dense(1, activation="linear")(z)
    
    # The model will accept the inputs of the all branches and
    # then output a single value
    model = Model(inputs=branches_in, outputs=z)
    model.summary()
    
    # Compile the model using mean absolute percentage error as the loss,
    # implying that we seek to minimize the absolute percentage difference
    # between predicted ddG and the actual ddG 
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_squared_error", optimizer=opt)


    #Preprocess data
    feature_train_set = dict()
    for samples in X_total_train:
        for i, feat in enumerate(FeatNames):
            if feat in feature_train_set:
                feature_train_set[feat].append(samples[:,i])
            else:
                feature_train_set[feat] = [samples[:,i]]
    
    feature_test_set = dict()
    for samples in X_total_test:
        for i, feat in enumerate(FeatNames):
            if feat in feature_test_set:
                feature_test_set[feat].append(samples[:,i])
            else:
                feature_test_set[feat] = [samples[:,i]]
    
    
    feature_train_set = [np.array(feature_train_set[feat]) for feat in FeatNames]
    feature_test_set = [np.array(feature_test_set[feat]) for feat in FeatNames]
    
    print(len(feature_train_set[0]))
    
    # Train the model
    print("Keras: training ...")
    history = model.fit(
	feature_train_set, y_total_train,
	validation_data=(feature_test_set, y_total_test),
	epochs=30, batch_size=4)
    
    # make predictions on the testing data
    print("Keras: Predicting " + source + "...")
    preds = model.predict(feature_test_set)


    diff = preds.flatten() - y_ddg_total_test
    percentDiff = (diff / y_ddg_total_test) * 100
    absPercentDiff = np.abs(percentDiff)
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    
    #print(percentDiff, absPercentDiff, mean, std)
    
    # Plot history
    plt.figure(figsize=(10,7))
    plt.plot(history.history['loss'], label='training data')
    plt.plot(history.history['val_loss'], label='validation data')
    plt.title(source + ' prediction')
    plt.ylabel('Error')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(res_path,'training_report_'+source+'_'+mode))




def DoKerasMagic2(X_group_train, X_group_test, y_group_train, y_group_test, FeatNames_group, source, mode):
    print('\n\n')
    print("##############################################")
    print("############# Keras-Training 2 ###############")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
    
    print(len(X_group_train))
    print(len(y_group_train))
    
    n_bkmdl = 30
    branches_in = []
    branches_out = []
    
    # Each branch operates on each feature (each input vector belongs to a feature)
    #total_out_dim = 0
    for gr, gr_len in FeatNames_groups.items():
        out_dim = int(np.floor(gr_len / n_bkmdl))
        inputL = Input(shape=(gr_len,))
        x = None
        x = Dense(out_dim, activation="relu")(inputL)
        #x = Dense(out_dim, activation="relu")(x)
        x = Model(inputs=inputL, outputs=x)
        branches_in.append(x.input)
        branches_out.append(x.output)
        #total_out_dim += out_dim
        
    # Combine the output of all branches    
    combined = concatenate(branches_out)
    
    # Apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(50, activation="relu")(combined)
    #z = Dense(100, activation="relu")(z)
    z = Dense(8, activation="relu")(z)
    z = Dense(1, activation="linear")(z)
    
    # The model will accept the inputs of the all branches and
    # then output a single value
    model = Model(inputs=branches_in, outputs=z)
    model.summary()
    
    # Compile the model using mean absolute percentage error as the loss,
    # implying that we seek to minimize the absolute percentage difference
    # between predicted ddG and the actual ddG 
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_squared_error", optimizer=opt)


    #Preprocess data
    group_train_set = dict()
    for sample in X_group_train:
        for i, gr_feat in enumerate(sample):
            if i in group_train_set:
                group_train_set[i].append(gr_feat)
            else:
                group_train_set[i] = [gr_feat]
    
    group_test_set = dict()
    for sample in X_group_test:
        for i, gr_feat in enumerate(sample):
            if i in group_test_set:
                group_test_set[i].append(gr_feat)
            else:
                group_test_set[i] = [gr_feat]
    
    
    group_train_set = [np.array(group_train_set[i]) for i in group_train_set]
    group_test_set = [np.array(group_test_set[i]) for i in group_test_set]
    
    # Train the model
    print("Keras: training ...")
    history = model.fit(
	group_train_set, y_group_train,
	validation_data=(group_test_set, y_group_test),
	epochs=30, batch_size=4)
    
    # make predictions on the testing data
    print("Keras: Predicting " + source + "...")
    preds = model.predict(group_test_set)


    diff = preds.flatten() - y_ddg_total_test
    percentDiff = (diff / y_ddg_total_test) * 100
    absPercentDiff = np.abs(percentDiff)
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    
    #print(percentDiff, absPercentDiff, mean, std)
    
    # Plot history
    plt.figure(figsize=(10,7))
    plt.plot(history.history['loss'], label='training data')
    plt.plot(history.history['val_loss'], label='validation data')
    plt.title(source + ' prediction')
    plt.ylabel('Error')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(res_path,'training_group_report_'+source+'_'+mode))








#General boxplot analysis 1  
def DoBoxPlot1(data, xlabel, ylabel, xticks, title, filename):    
    #Boxplot
    plt.figure(figsize=(10,7))
    boxplotElements = plt.boxplot(data,
                                   sym = 'go', whis = 1.2,
                                   widths = [0.8]*len(data), positions = range(len(data)),
                                   patch_artist = True)
    
    #plt.gca().xaxis.set_ticklabels(mutt_list, rotation = 45)
    for element in boxplotElements['medians']:
        element.set_color('red')
        element.set_linewidth(2)
    for element in boxplotElements['boxes']:
        element.set_edgecolor('navy')
        element.set_facecolor((0,0,0,0))
        element.set_linewidth(2)
        #element.set_linestyle('dashed')
        element.set_fill(False)
        #element.set_hatch('/')
    for element in boxplotElements['whiskers']:
        element.set_color('purple')
        element.set_linewidth(2)
    for element in boxplotElements['caps']:
        element.set_color('black')
        
    plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.gca().set_axisbelow(True)
    
    #top = 15
    top = 0.85
    for tick in range(len(data)):
        plt.text(tick, top - (top*0.05), len(data[tick]),
             horizontalalignment='center', size='x-small', weight='bold',
             color='k')     
    
    #plt.ylim(-top, top)
    plt.ylim(-1, 1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.xticks(range(len(xticks)), xticks)
    #plt.legend()
    plt.savefig(filename, dpi=200)

#General displot analysis 1
def DoDistPlot1(feat_dict, xlabel, ylabel, filename):
    for feat in feat_dict.keys():
        try:
            plt.figure()
            for mtt,mtt_dist in feat_dict[feat].items():
                sns_plot = sns.distplot(mtt_dist, hist=False, rug=False)
                sns_plot.set(xlabel=xlabel, ylabel=ylabel, title=feat, xlim = (-1, 1))
                sns_fig = sns_plot.get_figure()
                sns_fig.savefig(filename + "_" + feat)
        except:
            continue
        
        
#General displot analysis 2
def DoDistPlot2(feat_dict, xlabel, ylabel, filename):
    """
    list_mean = []
    list_std = []
    for feat, std_data in feat_dict.items():
        list_mean += std_data['mean']
        list_std += std_data['std']
    min_mean, max_mean, min_std, max_std = \
    min(list_mean), max(list_mean), min(list_std), max(list_std)
    
    min_mean = -0.02
    min_std = -0.02
    """
    
    xmargin = 0.02
    for feat, std_data in feat_dict.items():
        try:
            plt.figure(figsize=fsize)
            sns_plot = sns.distplot(std_data['mean'], hist=True, rug=False)
            sns_plot.set(xlabel=xlabel+" (mean of stds)", ylabel=ylabel, title=feat+'_mean', xlim = (min(std_data['mean'])-xmargin, max(std_data['mean'])+xmargin))
            sns_fig = sns_plot.get_figure()
            sns_fig.savefig(filename + "_" + feat + '_mean')
            
            plt.figure(figsize=fsize)
            sns_plot = sns.distplot(std_data['std'], hist=True, rug=False)
            sns_plot.set(xlabel=xlabel+" (std of stds)", ylabel=ylabel, title=feat+'_std', xlim = (min(std_data['std'])-xmargin, max(std_data['std'])+xmargin))
            sns_fig = sns_plot.get_figure()
            sns_fig.savefig(filename + "_" + feat + '_std')
        except:
            continue

#General displot analysis 3
def DoDistPlot3(feat_dict, xlabel, ylabel, filename):
    """
    list_std = []
    for feat, std_data in feat_dict.items():
        list_std += std_data
    min_std, max_std = min(list_std), max(list_std)
    min_std = -0.02
    """
    xmargin = 0.02
    for feat, std_data in feat_dict.items():
        try:            
            plt.figure(figsize=fsize)
            sns_plot = sns.distplot(std_data, hist=True, rug=False)
            sns_plot.set(xlabel=xlabel, ylabel=ylabel, title=feat+'_std_ProtAllMT', xlim = (min(std_data)-xmargin, max(std_data)+xmargin))
            sns_fig = sns_plot.get_figure()
            sns_fig.savefig(filename + "_" + feat)
        except:
            continue

"""
def backrubFeatureAnalysis_tmp(self, name):
    for source in ['wt']:
        filename = self.res_path + 'Backrub' + '_' + name + '_results_lisa_total_' + source
        if not self.original:
            results_lisa_total = pd.read_csv(filename + '.csv', sep=';')
            results_lisa_total_xray = pd.read_csv(self.res_path + 'XRay_ALL_results_lisa_wt.csv', sep=';')
            score_vector_l = self.score_vector_lisa
        else:
            results_lisa_total = pd.read_csv(filename + '_original.csv', sep=';')
            results_lisa_total_xray = pd.read_csv(self.res_path + 'XRay_ALL_results_lisa_wt_original.csv', sep=';')
            score_vector_l = self.score_vector_lisa_original
        
        score_vector_l = score_vector_l[:-1]
        
        min_dict = {}
        max_dict = {}
        feat_dict = {}
        for feat in score_vector_l:
            list_feat_concat = results_lisa_total[feat].tolist() + results_lisa_total_xray[feat].tolist()
            min_dict[feat] = np.amin(list_feat_concat)
            max_dict[feat] = np.amax(list_feat_concat)
            feat_dict[feat] = {}
        
        #min_dict_xray = {}
        #max_dict_xray = {}
        #for feat in score_vector_l:
        #    min_dict_xray[feat] = np.amin(results_lisa_total_xray[feat].tolist())
        #    max_dict_xray[feat] = np.amax(results_lisa_total_xray[feat].tolist())
        
        
        samples_dataframe = results_lisa_total[self.inden_vector_skmp]
        wt_already_processed = []
        for index, row in samples_dataframe.iterrows():
            try:
                protein_complex = str(row['#Pdb'])
                mutate_complex = str(row['Mutation(s)_cleaned'])
                interaction_region = str(row['iMutation_Location(s)'])
                complex_type = str(row['Hold_out_type'])
                experimental_method = str(row['Method'])
                b_affine_wt = float(row['Affinity_wt'])
                b_affine_mt = float(row['Affinity_mut'])
                
                if protein_complex in wt_already_processed:
                    continue
                wt_already_processed.append(protein_complex)
                
                print("Processing complex: " + protein_complex + '\t source: ' + source)
                
                lisa_features = results_lisa_total[results_lisa_total['#Pdb'] == protein_complex]
                lisa_features_norm = [(lisa_features[x].tolist() - min_dict[x]) / (max_dict[x] - min_dict[x]) for x in score_vector_l]
                
                lisa_features_xray = results_lisa_total_xray[results_lisa_total_xray['#Pdb'] == protein_complex]
                lisa_features_xray = lisa_features_xray.iloc[0]
                lisa_features_xray_norm = [(lisa_features_xray[x] - min_dict[x]) / (max_dict[x] - min_dict[x]) for x in score_vector_l]
                
                lisa_features_subt = [lisa_features_norm[x] - lisa_features_xray_norm[x] for x in range(len(lisa_features_norm))]
                doBoxPlot1(lisa_features_subt, 'Features', 'df_'+source+'_xray', score_vector_l, protein_complex, self.res_path + protein_complex + "_" + source)
                
                for feat in score_vector_l: 
                    feat_dict[feat][protein_complex] = lisa_features_subt[score_vector_l.index(feat)]
                
            except Exception as e:                    
                print("Ouch! An exception happened while processing complex: " + protein_complex + ' source: ' + source + 
                      "\tException: " + str(e) + "\nMore information:\n" + traceback.format_exc())
                continue
        
        doDistPlot1(feat_dict, "Subtraction of normalized features", "Density function", filename)
"""

def GetDataDistribution(complex_data_blocks, FeatNames, Thr1, Thr2, display):
    print('\n\n')
    print("##############################################")
    print("############ Feature Distribution ############")
    print("##############################################")
    
    """
    This variable holds mean and std of an array of standard deviations. 
    Each array contains the std of 30 models, and belongs to a mutation.
    """
    feat_protein_summary_1 = {}
    
    """
    This variable holds std of each protein. All mutations and models together.
    """
    feat_protein_summary_2 = {}
    
    for feat in FeatNames:
        feat_protein_summary_1[feat] = {}
        feat_protein_summary_1[feat]['mean'] = []
        feat_protein_summary_1[feat]['std'] = []
        
        feat_protein_summary_2[feat] = []
        
    for protein_complex, protein_blocks in complex_data_blocks.items():
        feat_dict = {}
        for feat in FeatNames:
            feat_dict[feat] = {}
            st_dev = []
            prot_mutations = []
            for i, block in enumerate(protein_blocks):
                feat_dict[feat][i] = block[:,FeatNames.index(feat)]
                prot_mutations += list(feat_dict[feat][i])
                st_dev.append(np.std(feat_dict[feat][i]))
            feat_protein_summary_1[feat]['mean'].append(np.mean(st_dev))
            feat_protein_summary_1[feat]['std'].append(np.std(st_dev))
            feat_protein_summary_2[feat].append(np.std(prot_mutations))
        
        if display:
            filepath = os.path.join(res_path, protein_complex)
            if os.path.exists(filepath):
                shutil.rmtree(filepath)
            os.makedirs(filepath)
            filename = os.path.join(filepath, protein_complex)
            DoDistPlot1(feat_dict, "DlDlG normalized features", "Density function", filename)

    filepath = os.path.join(res_path, "DlDlG_summary_dist")
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
    os.makedirs(filepath)
    filename = os.path.join(filepath, "DlDlG_summary_dist")
    save_obj(feat_protein_summary_1, filename)
    DoDistPlot2(feat_protein_summary_1, "Standard deviation of DlDlG", "Density function", filename)
    
    filepath = os.path.join(res_path, "DlDlG_summary_dist_ProtAllMT")
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
    os.makedirs(filepath)
    filename = os.path.join(filepath, "DlDlG_summary_dist_ProtAllMT")
    save_obj(feat_protein_summary_2, filename)
    DoDistPlot3(feat_protein_summary_2, "Standard deviation of DlDlG", "Density function", filename)
        
    bad_variation_list = []
    good_variation_list = []
    for feat, value in feat_protein_summary_2.items():
        if np.mean(value) < Thr1:
            bad_variation_list.append(feat)
        else:
            good_variation_list.append(feat)
    for feat in good_variation_list:
        if np.mean(feat_protein_summary_1[feat]['mean']) > Thr2:
            bad_variation_list.append(feat)
    
    with open(filename+'_bad_variation_list', 'w') as f_handler:
        for feat in bad_variation_list: f_handler.write(feat+'\n')
        
    return bad_variation_list
            
def GetCorrelationMatrix(corrmatpath, dataX, dataY, FeatNames, is_total, source, mode, Thr1, display):
    print('\n\n')
    print("##############################################")
    print("############# Correlation Matrix #############")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
        
    if is_total:
        dataX_tmp = np.array([]).reshape(0, len(FeatNames))
        dataY_tmp = np.repeat(dataY, len(dataX[0])).reshape((len(dataY)*len(dataX[0]),1))
        for dX in dataX:
            dataX_tmp = np.concatenate((dataX_tmp, dX))
        data = np.concatenate((dataX_tmp, dataY_tmp), axis = 1)
        info = 'total'
    else:
        dataY_tmp = dataY.reshape((len(dataY),1))
        data = np.concatenate((dataX, dataY_tmp), axis = 1)
        info = 'mean'
    data_df = pd.DataFrame(data=data, columns = FeatNames+['target'])
    
    
    corrmatsubdir = os.path.join(corrmatpath, mode)
    if not os.path.exists(corrmatsubdir):
        os.makedirs(corrmatsubdir)
    
    filename = os.path.join(corrmatsubdir, 'cormat_' + source + '_' + mode + '_' + info + '_')
    

    corr_spearman = data_df.corr(method='spearman')
    corr_pearson = data_df.corr(method='pearson')
    
    
    corr_pearson_abs = corr_pearson.abs()
    # Select upper triangle of correlation matrix
    upper = corr_pearson_abs.where(np.triu(np.ones(corr_pearson_abs.shape), k=1).astype(np.bool))
    high_corr_list = [column for column in upper.columns if any(upper[column] > Thr1)]
    if not display:
        with open(filename+'high_corr_list','w') as f_handler:
            for x in high_corr_list: f_handler.write(x + "\n")
        upper.to_csv(filename+'high_corr_df', sep=';')
        return high_corr_list
    
    """
    corrplt_spearman = Corrplot(corr_spearman)
    corrplt_pearson = Corrplot(corr_pearson)

    plt.figure()
    corrplt_spearman.plot(figsize=fsize, rotation=90, upper='circle')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + 'spearman')
    
    plt.figure()
    corrplt_pearson.plot(figsize=fsize, rotation=90, upper='circle')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename + 'pearson')
    """
        
    # Generate a mask for the upper triangle
    mask_spearman = np.triu(np.ones_like(corr_spearman, dtype=np.bool))
    mask_pearson = np.triu(np.ones_like(corr_pearson, dtype=np.bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=fsize)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_spearman, mask=mask_spearman, cmap=cmap, vmin=-1, vmax=1, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(filename + 'spearman_sns')
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=fsize)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_pearson, mask=mask_pearson, cmap=cmap, vmin=-1, vmax=1, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(filename + 'pearson_sns')

    return high_corr_list

"""
dataX_xr,dataY_xr,FeatNames = FeatureVectorCreatorXRay('ALL')
X_xr_train, X_xr_test, y_xr_train, y_xr_test = train_test_split(dataX_xr, dataY_xr, test_size=0.3, random_state=0)
FeatureSelection(X_xr_train, X_xr_test, y_xr_train, y_xr_test, FeatNames, 'WILDT',3)
Performance(X_xr_train, X_xr_test, y_xr_train, y_xr_test, FeatNames, 'WILDT',3)
CrossValidation(dataX_xr, dataY_xr,3)
"""
 
for mode in [2]: #XXX
    if mode == 0:
        print("******************* FLEX ************************")
    if mode == 1:
        print("\n\n\n\n******************* LISA ************************")
    if mode == 2:
        print("\n\n\n\n******************* LISA+FLEX ************************")
    
    """ XXX
    dataX_wt,dataX_mt,dataX_ddg, \
    dataX_total_wt,dataX_total_mt,dataX_total_ddg, \
    _,_,_, \
    dataY_wt,dataY_mt,dataY_ddg, \
    FeatNames,FeatNames_groups, \
    complex_data_blocks, \
    _,_,_,_ = FeatureVectorCreator('ALL', mode, [], False, False)


    #Feature analysis
    var_thr1 = 5
    var_thr2 = 20
    corr_thr = 0.8
    display = False
    corrmatpath = os.path.join(res_path, 'CorrelationMatrix')
    if not os.path.exists(corrmatpath):
        os.makedirs(corrmatpath)
    
    GetCorrelationMatrix(corrmatpath, dataX_wt, dataY_wt, FeatNames, False, 'WILDT', mode, corr_thr, display)
    GetCorrelationMatrix(corrmatpath, dataX_mt, dataY_mt, FeatNames, False, 'MUTTD', mode, corr_thr, display)
    GetCorrelationMatrix(corrmatpath, dataX_ddg, dataY_ddg, FeatNames, False, 'DlDlG', mode, corr_thr, display)
    
    GetCorrelationMatrix(corrmatpath, dataX_total_wt, dataY_wt, FeatNames, True, 'WILDT', mode, corr_thr, display)
    GetCorrelationMatrix(corrmatpath, dataX_total_mt, dataY_mt, FeatNames, True, 'MUTTD', mode, corr_thr, display)
    high_corr_list = GetCorrelationMatrix(corrmatpath, dataX_total_ddg, dataY_ddg, FeatNames, True, 'DlDlG', mode, corr_thr, display)    
    
    bad_variation_list = GetDataDistribution(complex_data_blocks, FeatNames, var_thr1, var_thr2, display)
    
    feature_drop_list = list(set(bad_variation_list + high_corr_list))
    
    print("\n\nfeature_drop_list:")
    print(feature_drop_list)
    print("\n\n")
    """
    feature_drop_list = [] #XXX
    
    
    dataX_wt,dataX_mt,dataX_ddg, \
    dataX_total_wt,dataX_total_mt,dataX_total_ddg, \
    dataX_total_group_wt,dataX_total_group_mt,dataX_total_group_ddg, \
    dataY_wt,dataY_mt,dataY_ddg, \
    FeatNames,FeatNames_groups, \
    complex_data_blocks, \
    _,_,_,_ = FeatureVectorCreator('ALL', mode, feature_drop_list, False, False) #XXX pre_process=True

    """
    dataX_gl = np.concatenate((dataX_wt,dataX_mt))
    dataY_gl = np.concatenate((dataY_wt,dataY_mt))
    
    X_wt_train, X_wt_test, y_wt_train, y_wt_test = train_test_split(dataX_wt, dataY_wt, test_size=0.2, random_state=0, shuffle=True)
    X_mt_train, X_mt_test, y_mt_train, y_mt_test = train_test_split(dataX_mt, dataY_mt, test_size=0.2, random_state=0, shuffle=True)
    X_ddg_train, X_ddg_test, y_ddg_train, y_ddg_test = train_test_split(dataX_ddg, dataY_ddg, test_size=0.2, random_state=0, shuffle=True)
    X_gl_train, X_gl_test, y_gl_train, y_gl_test = train_test_split(dataX_gl, dataY_gl, test_size=0.2, random_state=0, shuffle=True)
    """
    
    X_wt_total_train, X_wt_total_test, y_wt_total_train, y_wt_total_test = train_test_split(dataX_total_wt, dataY_wt, test_size=0.2, random_state=0, shuffle=True)
    X_mt_total_train, X_mt_total_test, y_mt_total_train, y_mt_total_test = train_test_split(dataX_total_mt, dataY_mt, test_size=0.2, random_state=0, shuffle=True)
    X_ddg_total_train, X_ddg_total_test, y_ddg_total_train, y_ddg_total_test = train_test_split(dataX_total_ddg, dataY_ddg, test_size=0.2, random_state=0, shuffle=True)

    X_wt_total_group_train, X_wt_total_group_test, y_wt_total_group_train, y_wt_total_group_test = train_test_split(dataX_total_group_wt, dataY_wt, test_size=0.2, random_state=0, shuffle=True)
    X_mt_total_group_train, X_mt_total_group_test, y_mt_total_group_train, y_mt_total_group_test = train_test_split(dataX_total_group_mt, dataY_mt, test_size=0.2, random_state=0, shuffle=True)
    X_ddg_total_group_train, X_ddg_total_group_test, y_ddg_total_group_train, y_ddg_total_group_test = train_test_split(dataX_total_group_ddg, dataY_ddg, test_size=0.2, random_state=0, shuffle=True)


    
    DoKerasMagic(X_ddg_total_train, X_ddg_total_test, y_ddg_total_train, y_ddg_total_test, FeatNames, 'DlDlG', mode)
    DoKerasMagic(X_wt_total_train, X_wt_total_test, y_wt_total_train, y_wt_total_test, FeatNames, 'WILDT', mode)
    DoKerasMagic(X_mt_total_train, X_mt_total_test, y_mt_total_train, y_mt_total_test, FeatNames, 'MUTTD', mode)
    
    
    
    DoKerasMagic2(X_ddg_total_group_train, X_ddg_total_group_test, y_ddg_total_group_train, y_ddg_total_group_test, FeatNames_groups, 'DlDlG', mode)
    DoKerasMagic2(X_wt_total_group_train, X_wt_total_group_test, y_wt_total_group_train, y_wt_total_group_test, FeatNames_groups, 'WILDT', mode)
    DoKerasMagic2(X_mt_total_group_train, X_mt_total_group_test, y_mt_total_group_train, y_mt_total_group_test, FeatNames_groups, 'MUTTD', mode)
    
    
    
    
"""
    print("\n\n\n\n******************* Processing WILD-TYPE ************************")
    FeatureSelection(X_wt_train, X_wt_test, y_wt_train, y_wt_test, FeatNames, 'WILDT',mode)
    Performance(X_wt_train, X_wt_test, y_wt_train, y_wt_test, FeatNames, 'WILDT',mode)
    CrossValidation(dataX_wt, dataY_wt,mode)

    print("\n\n\n\n******************* Processing MUTATED ************************")
    FeatureSelection(X_mt_train, X_mt_test, y_mt_train, y_mt_test, FeatNames, 'MUTTD',mode)
    Performance(X_mt_train, X_mt_test, y_mt_train, y_mt_test, FeatNames, 'MUTTD',mode)
    CrossValidation(dataX_mt, dataY_mt,mode)

    print("\n\n\n\n******************* Processing DDG ************************")
    FeatureSelection(X_ddg_train, X_ddg_test, y_ddg_train, y_ddg_test, FeatNames, 'DlDlG',mode)
    Performance(X_ddg_train, X_ddg_test, y_ddg_train, y_ddg_test, FeatNames, 'DlDlG',mode)
    CrossValidation(dataX_ddg, dataY_ddg,mode)
    
    print("\n\n\n\n******************* Processing Global ************************")
    FeatureSelection(X_gl_train, X_gl_test, y_gl_train, y_gl_test, FeatNames, 'Global',mode)
    Performance(X_gl_train, X_gl_test, y_gl_train, y_gl_test, FeatNames, 'Global',mode)
    CrossValidation(dataX_gl, dataY_gl,mode)

BestModelEvaluation('blindtest')
"""
