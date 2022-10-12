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

score_vector_lisa = ["V39","V40","V41","V42","V43","V44","V45","V46","V47","V48","V49","V50","V51","V52","V53","V54",
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

score_vector_flex = ['fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_rep','fa_sol','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','nstruct','omega','p_aa_pp','pro_close','rama','ref','total_score','yhh_planarity'] 
#score_vector_flex = ['fa_atr','fa_elec','fa_rep','fa_sol','hbond_bb_sc','hbond_lr_bb','hbond_sc']
#score_vector_flex = ['fa_atr','fa_elec','fa_intra_rep','fa_rep','fa_sol'] 

score_vector_groups = {'lg1': score_vector_lisa[:168], 
                       'lg2': score_vector_lisa[168:176], 
                       'lg3': score_vector_lisa[176:],
                       'fg1': score_vector_flex}

res_path = 'Results/'    

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

def PreprocessData(X_train, X_test):
	# Performing min-max scaling each continuous feature column to the range [0, 1]
	cnts = MinMaxScaler()
	X_train_scaled = cnts.fit_transform(X_train)
	X_test_scaled = cnts.transform(X_test)
	return X_train_scaled, X_test_scaled
 
def FetchData(name, mode):
    source = "Backrub"
    output_files = os.path.join(res_path, source + '_' + name)
    results_flex_wt = pd.read_csv(output_files + '_results_flex_wt.csv', sep=';').reset_index(drop=True)
    results_flex_mt = pd.read_csv(output_files + '_results_flex_mt.csv', sep=';').reset_index(drop=True)
    
    results_lisa_wt = pd.read_csv(output_files + '_results_lisa_wt.csv', sep=';').reset_index(drop=True)
    results_lisa_mt = pd.read_csv(output_files + '_results_lisa_mt.csv', sep=';').reset_index(drop=True)
    
    
    dataY_mt = results_flex_mt['Affinity_mut']
    dataY_wt = results_flex_wt['Affinity_wt']
    dataY = pd.concat([dataY_mt,dataY_wt]).reset_index(drop=True)
    
    if mode == 0:
        dataX_wt = results_flex_wt[score_vector_flex]
        dataX_mt = results_flex_mt[score_vector_flex]        
    if mode == 1:
        dataX_wt = results_lisa_wt[score_vector_lisa]
        dataX_mt = results_lisa_mt[score_vector_lisa]
    if mode == 2:
        dataX_wt = pd.concat([results_flex_wt[score_vector_flex],results_lisa_wt[score_vector_lisa]],axis=1)
        dataX_mt = pd.concat([results_flex_mt[score_vector_flex],results_lisa_mt[score_vector_lisa]],axis=1)
    dataX = pd.concat([dataX_mt, dataX_wt]).reset_index(drop=True)
    
    return dataX, dataY           

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


def DoKerasMagic(X_train, X_test, y_train, y_test, f_names, f_groups, source, mode):
    print('\n\n')
    print("##############################################")
    print("############# Keras-Training #################")
    print("##############################################")
          
    if mode == 0:
        mode = "Flex"
    elif mode == 1:
        mode = "Lisa"
    elif mode == 2:
        mode = "FlexLisa"
    
    branches_in = []
    branches_out = []

    indexes = {}
    for gr, gr_feat in f_groups.items():
        indexes[gr] = [f_names.index(f) for f in gr_feat if f in f_names]
    
    # Each branch operates on each feature (each input vector belongs to a feature)
    for gr_feat in indexes.values():
        if gr_feat == []:
            continue
        gr_len = len(gr_feat)
        out_dim = int(np.floor(gr_len / 2))
        inputL = Input(shape=(gr_len,))
        x = None
        x = Dense(gr_len, activation="relu")(inputL)
        x = Dense(out_dim, activation="relu")(x)
        x = Model(inputs=inputL, outputs=x)
        branches_in.append(x.input)
        branches_out.append(x.output)
        
    # Combine the output of all branches    
    combined = concatenate(branches_out)
    
    # Apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(60, activation="relu")(combined)
    z = Dense(20, activation="relu")(z)
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
    
    train_set = [X_train[:,gr_feat] for gr_feat in indexes.values() if gr_feat != []]
    test_set = [X_test[:,gr_feat] for gr_feat in indexes.values() if gr_feat != []]
    
    # Train the model
    print("Keras: training ...")
    history = model.fit(
	train_set, y_train,
	validation_data=(test_set, y_test),
	epochs=150, batch_size=10)
    
    # make predictions on the testing data
    print("Keras: Predicting " + source + "...")
    preds = model.predict(test_set)


    diff = preds.flatten() - y_test
    percentDiff = (diff / y_test) * 100
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





def DoKerasMagic2(X_train, X_test, y_train, y_test, f_names, source, mode):
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
    
    
    inputL = Input(shape=(len(f_names),))
            
    # Apply a FC layer and then a regression prediction
    z = Dense(10, activation="relu")(inputL)
    z = Dense(8, activation="relu")(z)
    z = Dense(1, activation="linear")(z)
    
    # The model will accept the inputs of the all branches and
    # then output a single value
    model = Model(inputs=inputL, outputs=z)
    model.summary()
    
    # Compile the model using mean absolute percentage error as the loss,
    # implying that we seek to minimize the absolute percentage difference
    # between predicted ddG and the actual ddG 
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_squared_error", optimizer=opt)
        
    # Train the model
    print("Keras: training ...")
    history = model.fit(
	X_train, y_train,
	validation_data=(X_test, y_test),
	epochs=150, batch_size=10)
    
    # make predictions on the testing data
    print("Keras: Predicting " + source + "...")
    preds = model.predict(X_test)


    diff = preds.flatten() - y_test
    percentDiff = (diff / y_test) * 100
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


def GetCorrelationMatrix(corrmatpath, dataX, FeatNames, source, mode, Thr1, display):
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
    info = 'mean'
    
    data_df = dataX
    
    corrmatsubdir = os.path.join(corrmatpath, mode)
    if os.path.exists(corrmatsubdir):
        shutil.rmtree(corrmatsubdir)
    os.makedirs(corrmatsubdir)
    
    filename = os.path.join(corrmatsubdir, 'cormat_' + source + '_' + mode + '_' + info + '_')
    
    corr_spearman = data_df.corr(method='spearman')
    corr_pearson = data_df.corr(method='pearson')
    
    corr_pearson_abs = corr_pearson.abs()
    # Select upper triangle of correlation matrix
    upper = corr_pearson_abs.where(np.triu(np.ones(corr_pearson_abs.shape), k=1).astype(np.bool))
    high_corr_list = [column for column in upper.columns if any(upper[column] > Thr1)]

    with open(filename+'high_corr_list','w') as f_handler:
        for x in high_corr_list: f_handler.write(x + "\n")
    upper.to_csv(filename+'high_corr_df', sep=';')
    
    if not display:
        return high_corr_list
     
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


for mode in [0,1,2]:
    if mode == 0:
        print("******************* FLEX ************************")
    if mode == 1:
        print("\n\n\n\n******************* LISA ************************")
    if mode == 2:
        print("\n\n\n\n******************* LISA+FLEX ************************")
    
    
    dataX, dataY = FetchData('ALL', mode)

    FeatNames = dataX.columns.to_list()    
    
    #Feature analysis
    corr_thr = 0.93
    display = True
    corrmatpath = os.path.join(res_path, 'CorrelationMatrix')
    if not os.path.exists(corrmatpath):
        os.makedirs(corrmatpath)
    
    high_corr_list = GetCorrelationMatrix(corrmatpath, dataX, FeatNames, 'Global', mode, corr_thr, display)
    
    dataX = dataX.drop(high_corr_list, axis=1)
    FeatNames = dataX.columns.to_list()
    
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=0, shuffle=True)
     
    y_max = y_train.max()
    y_train = y_train / y_max
    y_test = y_test / y_max
    X_train, X_test = PreprocessData(X_train, X_test)
    
    if mode == 2:
        DoKerasMagic(X_train, X_test, y_train, y_test, FeatNames, score_vector_groups, 'Global', mode)
        
    print("\n\n\n\n******************* Processing Global ************************")
    FeatureSelection(X_train, X_test, y_train, y_test, FeatNames, 'Global', mode)
    Performance(X_train, X_test, y_train, y_test, FeatNames, 'Global', mode)
    CrossValidation(dataX.to_numpy(), dataY.to_numpy(), mode)


bestfeatures = ['fa_atr','fa_elec','fa_rep','fa_sol','hbond_bb_sc','hbond_lr_bb','hbond_sc','total_score',
                'V48','V50','V134','V175','V207','V208','V209','V211','V212','V214','nis1','nis2','nis3']

mode = 2
dataX, dataY = FetchData('ALL', mode)
dataX = dataX[bestfeatures]
FeatNames = bestfeatures
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=0, shuffle=True)
y_max = y_train.max()
y_train = y_train / y_max
y_test = y_test / y_max
X_train, X_test = PreprocessData(X_train, X_test)
DoKerasMagic2(X_train, X_test, y_train, y_test, FeatNames, 'BestFeat', mode)