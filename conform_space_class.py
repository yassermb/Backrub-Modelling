#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:43:13 2019

@author: mohseni
"""

import sys
import numpy as np
import pandas as pd
#from pymol import cmd #Only for mutagenesis
from pyrosetta import *
from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.protocols.scoring import Interface
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.utility import *
#from Bio.PDB import *
#import nglview as nv
import os,warnings,shutil
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import time
import traceback

#sys.setrecursionlimit(1500)
warnings.filterwarnings('ignore')


#Global variables
INTBuilder_path = "/home/mohseni/mySoftwares/INTBuilder/INTBuilder"

class ConformationalModeling:
    def __init__(self):
        self.pdb_path = 'SKEMPI2_PDBs/PDBs/'
        self.pdb_path_AffinityB2 = 'AffinityBenchmark/V2/Affinity_Benchmark/'
        self.lisa_path = 'LISA-output/p2/'
        self.res_path = 'Results/'
        self.tmp_path = 'Temp/'
        self.flex_ddg_input = self.tmp_path + "inputs/"
        self.flex_ddg_output = self.tmp_path + "output/"
        self.flex_ddg_output_raw = self.tmp_path + "output_raw/"
        self.flex_ddg_analysis = self.tmp_path + "analysis_output/"
        self.flex_ddg_input_bu = self.tmp_path + "inputs_bu/"
        self.flex_ddg_output_bu = self.tmp_path + "output_bu/"
        self.raw_ref_struct = self.tmp_path + "raw_ref_struct/"
        #self.ref_db_path = "../../../myStage/FinalExperiment/Temp/output_bu"
        
        
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
            
            
        self.protein_letters_1to3 = {
            'A': 'ala', 'C': 'cys', 'D': 'asp',
            'E': 'glu', 'F': 'phe', 'G': 'gly', 'H': 'his',
            'I': 'ile', 'K': 'lys', 'L': 'leu', 'M': 'met',
            'N': 'asn', 'P': 'pro', 'Q': 'gln', 'R': 'arg',
            'S': 'ser', 'T': 'thr', 'V': 'val', 'W': 'trp',
            'Y': 'tyr'}
            
        self.protein_letters_3to1 = {v: k for k, v in self.protein_letters_1to3.items()}
        
        self.aminoacid_classes = {
                'nonpolar': ['A','V','L','I','P','F','W','M'],
                'polar': ['G','S','T','C','Y','N','Q'],
                'positive': ['K','R','H'],
                'negative': ['D','E']}
        
        
        self.inden_vector_skmp = ['#Pdb', 'Mutation(s)_cleaned', 'iMutation_Location(s)', 'Hold_out_type', 'Method', 'Affinity_wt', 'Affinity_mut']
        self.inden_vector_skmp_wt = ['#Pdb', 'Hold_out_type', 'Method', 'Affinity_wt']                          
        self.inden_vector_affB = ['Complex PDB', 'Type', 'Method', 'dG']
        self.score_vector_lisa = ['V106','V46','V202','V208','V107','V114','V154','V207','V69','nis2','b_affine']
        self.score_vector_lisa_original = ["V39","V40","V41","V42","V43","V44","V45","V46","V47","V48","V49","V50","V51","V52","V53","V54",
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
                                      "LogIntArea7","LogIntArea8","LogIntArea9","LogNumCon","b_affine"]
        self.original = False
        self.score_vector_flex = ['fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_rep','fa_sol','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','nstruct','omega','p_aa_pp','pro_close','rama','ref','total_score','yhh_planarity'] 
        
        #Load SKEMPI V 2.0
        self.dataframe_skempi = pd.read_csv('skempi_v2.csv', sep=';') 
        
        #Initial filtering based on the experimental methods for measuring the binding affinity
        self.dataframe_methods = self.dataframe_skempi[self.dataframe_skempi['Method'].isin(['ITC','SPR','SP','FL'])]
        
        
        #Isothermal Titration Calorimetry
        self.dataframe_itc = self.dataframe_skempi[self.dataframe_skempi['Method'] == 'ITC']
        #Surface Plasmon Resonance
        self.dataframe_spr = self.dataframe_skempi[self.dataframe_skempi['Method'] == 'SPR']
        #Spectroscopic inhibition assay
        self.dataframe_isp = self.dataframe_skempi[self.dataframe_skempi['Method'] == 'IASP']
        #Other spectroscopic method
        self.dataframe_gsp = self.dataframe_skempi[self.dataframe_skempi['Method'] == 'SP']
        #Fluorescence
        self.dataframe_flr = self.dataframe_skempi[self.dataframe_skempi['Method'] == 'FL']
        
        
        #Filtering based on the complex type
        self.dataframe_PrPI = self.dataframe_methods[self.dataframe_methods['Hold_out_type'] == 'Pr/PI']
        self.dataframe_ABAG = self.dataframe_methods[self.dataframe_methods['Hold_out_type'] == 'AB/AG']
        self.dataframe_TCRpMHC = self.dataframe_methods[self.dataframe_methods['Hold_out_type'] == 'TCR/pMHC']
        self.dataframe_typeUnknown = self.dataframe_methods[self.dataframe_methods['Hold_out_type'].isnull()]
        
        self.dataframe_itc_PrPI = self.dataframe_itc[self.dataframe_itc['Hold_out_type'] == 'Pr/PI']
        self.dataframe_itc_ABAG = self.dataframe_itc[self.dataframe_itc['Hold_out_type'] == 'AB/AG']
        self.dataframe_itc_TCRpMHC = self.dataframe_itc[self.dataframe_itc['Hold_out_type'] == 'TCR/pMHC']
        
        self.dataframe_spr_PrPI = self.dataframe_spr[self.dataframe_spr['Hold_out_type'] == 'Pr/PI']
        self.dataframe_spr_ABAG = self.dataframe_spr[self.dataframe_spr['Hold_out_type'] == 'AB/AG']
        self.dataframe_spr_TCRpMHC = self.dataframe_spr[self.dataframe_spr['Hold_out_type'] == 'TCR/pMHC']
        
        self.dataframe_gsp_PrPI = self.dataframe_gsp[self.dataframe_gsp['Hold_out_type'] == 'Pr/PI']
        self.dataframe_gsp_ABAG = self.dataframe_gsp[self.dataframe_gsp['Hold_out_type'] == 'AB/AG']
        self.dataframe_gsp_TCRpMHC = self.dataframe_gsp[self.dataframe_gsp['Hold_out_type'] == 'TCR/pMHC']
        
        self.dataframe_flr_PrPI = self.dataframe_flr[self.dataframe_flr['Hold_out_type'] == 'Pr/PI']
        self.dataframe_flr_ABAG = self.dataframe_flr[self.dataframe_flr['Hold_out_type'] == 'AB/AG']
        self.dataframe_flr_TCRpMHC = self.dataframe_flr[self.dataframe_flr['Hold_out_type'] == 'TCR/pMHC']
        
        #Filtering based on the region of mutation
        self.dataframe_COR = self.dataframe_methods[self.dataframe_methods['iMutation_Location(s)'] == 'COR']
        self.dataframe_SUP = self.dataframe_methods[self.dataframe_methods['iMutation_Location(s)'] == 'SUP']
        self.dataframe_RIM = self.dataframe_methods[self.dataframe_methods['iMutation_Location(s)'] == 'RIM']
        self.dataframe_INT = self.dataframe_methods[self.dataframe_methods['iMutation_Location(s)'] == 'INT']
        self.dataframe_SUR = self.dataframe_methods[self.dataframe_methods['iMutation_Location(s)'] == 'SUR']
        
        """
        Samples of the reference database
        """
        #self.reference_database = os.listdir(self.ref_db_path)
        
        pyrosetta.init()
    
        
    """
    A function to convert the dissociation constant to Delta G 
    """
    @staticmethod
    def Kd2DeltaG(Kd):
        return (8.314/4184)*(273.15+25)*np.log(Kd)

    """
    Distribution of SKEMPI ddG
    """
    def AnalyzeSKEMPITarget(self):
        import matplotlib.pyplot as plt
        dataframe = self.dataframe_methods
        ddg_list = []
        for index, row in dataframe.iterrows():
            try:
                protein_complex = str(row['#Pdb'])
                mutate_complex = str(row['Mutation(s)_cleaned'])
                interaction_region = str(row['iMutation_Location(s)'])
                complex_type = str(row['Hold_out_type'])
                experimental_method = str(row['Method'])
                b_affine_wt = str(row['Affinity_wt (M)'])
                b_affine_mt = str(row['Affinity_mut (M)'])
                
                b_affine_wt = self.Kd2DeltaG(float(b_affine_wt.replace("<","").replace(">","")))
                b_affine_mt = self.Kd2DeltaG(float(b_affine_mt.replace("<","").replace(">","")))
                ddg_list.append(b_affine_mt-b_affine_wt)
            except:
                continue
        plt.figure(figsize=(12,10))
        sns.set(font_scale=1.15)
        sns_plot = sns.distplot(ddg_list, hist=True, rug=False)
        sns_plot.set(xlabel="$\Delta\Delta$G", ylabel="Distribution", title="Distribution of $\Delta\Delta$G")
        sns_fig = sns_plot.get_figure()
        sns_fig.savefig('DistSKEMPIDDG')
        
        
    """
    This function finds the interaction surface in a protein complex using pyrosetta and saves it into rosetta resfile
    """
    def FindTheInteractionSurface(self, pdb_code, chains1, chains2):
        in_path = self.flex_ddg_input + pdb_code + '/'
        pdb_file = in_path + pdb_code + ".pdb"
        inter_resfile = in_path + "interface_residues.resfile"
        cleanATOM(pdb_file)
        pose = pose_from_pdb(in_path + pdb_code + ".clean.pdb")
        
        jump_num = 1
        setup_foldtree(pose, chains1 + "_" + chains2, Vector1([jump_num]))
        #print(pose.fold_tree())
        
        interface = Interface(jump_num)
        interface.distance(8.0)
        interface.calculate(pose) 
        inter_residues_tmp = list(interface.pair_list())
        inter_residues = list(inter_residues_tmp[0]) + list(inter_residues_tmp[1]) 
        #interface.print(pose)
        
        if inter_residues == []: #XXX I don't know why it happens for some complexes! It must be resolved!
            inter_residues = list(range(1,len(pose.sequence())))
        
        print(inter_residues)
        
        #Create resfile
        with open(inter_resfile,"w") as outfile:
            outfile.write("NATAA\n")
            outfile.write("start\n")
            for resi in inter_residues:
                chain_name = (chains1+chains2)[pose.chain(resi)-1]
                res_name = self.protein_letters_3to1[str(pose.aa(resi))[-3:].lower()]
                offset = pose.chain_end(pose.chain(resi)-1) if pose.chain(resi) > 1 else 0 
                res_pos = str(resi-offset)
                outfile.write(res_pos + ' ' + chain_name + ' PIKAA ' + res_name + "\n")

    def AffinityPrediction(self, pdb_file, chains1, chains2, separate_partners):
        if separate_partners:
            #Separate proteins from complex
            with open(pdb_file) as infile:
                with open(self.tmp_path + "p1.pdb",'w') as outfile:        
                    previousChainLabel = '-'
                    for line in infile:
                        if line[77:78] == 'H': #Remove the hydrogen atoms introduced by Backrub
                            continue
                        if line[:6] == 'SSBOND':
                            continue
                        if "#BEGIN_POSE_ENERGIES_TABLE" in line: #For the structures extracted from rosetta backrub
                            break
                        if line[:6] == 'EXPDTA' or line[:6] == 'REMARK':
                            continue
                        if line[:3] == 'END':
                            outfile.write(line)
                        elif line[:3] == 'TER' and len(line) < 10: #There is TER line without specifying its chain!
                            if previousChainLabel in chains1:
                                outfile.write(line)
                        elif line[21] in chains1:
                            previousChainLabel = line[21]
                            outfile.write(line)
                            
            with open(pdb_file) as infile:
                with open(self.tmp_path + "p2.pdb",'w') as outfile:        
                    previousChainLabel = '-'
                    for line in infile:
                        if line[77:78] == 'H': #Remove the hydrogen atoms introduced by Backrub
                            continue
                        if line[:6] == 'SSBOND':
                            continue
                        if "#BEGIN_POSE_ENERGIES_TABLE" in line: #For the structures extracted from rosetta backrub
                            break
                        if line[:6] == 'EXPDTA' or line[:6] == 'REMARK':
                            continue
                        if line[:3] == 'END':
                            outfile.write(line)
                        elif line[:3] == 'TER' and len(line) < 10: #There is TER line without specifying its chain!
                            if previousChainLabel in chains2:
                                outfile.write(line)
                        elif line[21] in chains2:
                            previousChainLabel = line[21]
                            outfile.write(line)
        else:
            pdb_code = pdb_file
            command = "cp " + self.pdb_path_AffinityB2 + pdb_code + "_l_b.pdb" + " " +  self.tmp_path + "p1.pdb"
            os.system(command)
            command = "cp " + self.pdb_path_AffinityB2 + pdb_code + "_r_b.pdb" + " " +  self.tmp_path + "p2.pdb"
            os.system(command)
                    
        #Apply LISA        
        #command = "python2 ../Softwares/LISA_1.0/LISA/LISA-1.0.py " + tmp_path + "p1.pdb " + tmp_path +  "p2.pdb 8"
        command = "python2 ../../myStage/Softwares/LISA_1.1/LISA/LISA-1.0.py " + self.tmp_path + "p1.pdb " + self.tmp_path +  "p2.pdb 35"
        os.system(command)
        command = "Rscript " + self.lisa_path + "LISA_bindingAff.R"
        os.system(command)
        
        #Get LISA's output
        with open(self.lisa_path + "LISA_bindingAff.txt") as f_handler:
            b_affine_predict = float(f_handler.readlines()[1].split(' ')[1])
        
        #Get LISA's features
        lisa_feature_results = pd.read_csv(self.lisa_path + 'results.txt', sep=',')
        lisa_feature_results = lisa_feature_results.drop(lisa_feature_results.columns[0], axis=1)
        #lisa_feature_results = lisa_feature_results.iloc[0].tolist()[1:]
        #lisa_feature_results = [float(x) for x in lisa_feature_results]
        
        lisa_feature_results_original = pd.read_csv(self.lisa_path + 'fullResults.txt', sep=',')
        lisa_feature_results_original = lisa_feature_results_original.drop(lisa_feature_results_original.columns[0], axis=1)
            
        lisa_results = pd.concat([lisa_feature_results,
                                  pd.DataFrame([[b_affine_predict]],columns=['b_affine'])], axis=1)
    
        lisa_results_original = pd.concat([lisa_feature_results_original,
                                           pd.DataFrame([[b_affine_predict]],columns=['b_affine'])], axis=1)
       
        os.remove(self.tmp_path + "p1.pdb")
        os.remove(self.tmp_path + "p2.pdb")
        
        return lisa_results, lisa_results_original

    """
    A function to explore conformational space by applying local backrub sampling, local repacking, and global minimization.
    The local repacking and backrub sampling is performed on the interaction interface (and 8 A around it) of the protein complex.
    references:
        1) Flex ddG: Rosetta Ensemble-Based Estimation of Changes inProtein−Protein Binding Affinity upon Mutation
        2) Role of conformational sampling in computing mutation-induced changes in protein structure and stability
        3) Backrub-Like Backbone Simulation Recapitulates Natural Protein Conformational Variability and Improves Mutant Side-Chain Prediction
    """
    
    """
    STEP 1: Generate raw reference backrub structures and WT backrub models.
    They are generated once per complex!
    """
    def ExploreConformationalSpace_Step1(self, name, number, backrub_steps):
        
        dataframe = self.dataframe_methods

        if not os.path.exists(self.raw_ref_struct):
            os.makedirs(self.raw_ref_struct)
           
        samples_already_processed = []
        samples_exist_output = os.listdir(self.raw_ref_struct) #These are the samples that already exist in the raw_ref_struct
        #It is helpful to resume the process and continue with the rest of samples!
        for current_sample in samples_exist_output:
            samples_already_processed.append(current_sample)
            
        count = 0
        for index, row in dataframe.iterrows():
            if count > number:
                break
            try:
                protein_complex = str(row['#Pdb'])
                #mutate_complex = str(row['Mutation(s)_cleaned'])
                #interaction_region = str(row['iMutation_Location(s)'])
                #complex_type = str(row['Hold_out_type'])
                #experimental_method = str(row['Method'])
                #b_affine_wt = str(row['Affinity_wt (M)'])
                #b_affine_mt = str(row['Affinity_mut (M)'])
                
                current_sample = protein_complex
                folder_name = protein_complex
                
                print("\n\nProcessing sample: index=" + str(index) + "\tComplex=" + protein_complex)
                
                if current_sample in samples_already_processed:
                    print("This sample is already processed! We also avoid processing samples in which the mutation happens on the sampe place of the same complex!")
                    continue
                
                samples_already_processed.append(current_sample)          
                            
                comp_info = protein_complex.split('_')
                pdb_code, chains1, chains2 = comp_info[0], comp_info[1], comp_info[2]
                pdb_file = self.pdb_path + pdb_code + '.pdb'
                
                if os.path.exists(self.flex_ddg_input):
                    shutil.rmtree(self.flex_ddg_input)
                os.makedirs(self.flex_ddg_input)
                
                if os.path.exists(self.flex_ddg_output):
                    shutil.rmtree(self.flex_ddg_output)
                os.makedirs(self.flex_ddg_output)
                            
                in_path = self.flex_ddg_input + pdb_code + '/'
                os.makedirs(in_path)
                shutil.copyfile(pdb_file,in_path + pdb_code + '.pdb')
                
                out_path = self.flex_ddg_output + pdb_code + '/'
                
                self.FindTheInteractionSurface(pdb_code, chains1, chains2)
                
                with open(in_path+"chains_to_move.txt", 'w') as fileH_chains:
                    ch_move = chains2
                    if len(chains2) > 1:
                        ch_move = ""
                        for c in chains2:
                            ch_move += c+','
                        ch_move = ch_move[:-1]
                    fileH_chains.write(ch_move)
                                
                print("\n\nSTEP 1\n\n")
                
                command = "python flex_ddg_run.py step1"
                os.system(command)
                
                #Extract structures created by local backrub sampling, local repacking, and global minimization processes
                command = "python flex_ddg_extract_structures.py " + self.flex_ddg_output
                os.system(command)
                #ATTENTION: there are useful information and scores for each residue in the extracted backrub structure/pdb!
                #Very nice detailed table!

                #Check if pdb files are extracted successfully (for example for the first model)
                if not os.path.isfile(out_path+"/01/wt_01_"+str(backrub_steps).zfill(5)+".pdb"): #This problem must be resolved properly!
                    raise Exception("The pdb files are not extracted!")
                
                #Save results of step 1
                #command = "cp -r " + out_path + " " + self.raw_ref_struct + folder_name + "/"      
                #os.system(command)
                print("\n\nSAVE: STEP1\n\n")
                shutil.copytree(out_path, os.path.join(self.raw_ref_struct, folder_name))
                shutil.copyfile(os.path.join(in_path, "chains_to_move.txt"), os.path.join(self.raw_ref_struct, folder_name, "chains_to_move.txt"))
                shutil.copyfile(os.path.join(in_path, "interface_residues.resfile"), os.path.join(self.raw_ref_struct, folder_name, "interface_residues.resfile"))
                
            except Exception as e:
                print("\nOuch! An exception happened while processing sample: index=" + str(index) + "\tComplex=" + protein_complex + 
                      "\tException: " + str(e) + "\nMore information:\n" + traceback.format_exc())
                continue
            count+=1
    
    """
    STEP 2: Generate mutant backrub models from raw reference backrub structures.
    They are generated per each new mutation on the protein complex!
    """
    def ExploreConformationalSpace_Step2(self, name, number, backrub_steps):
        
        dataframe = self.dataframe_methods
        
        if not os.path.exists(self.flex_ddg_output_bu):
            os.makedirs(self.flex_ddg_output_bu)
           
        samples_already_processed = [] #To avoid processing samples in which the mutation happens on the sampe place of the same complex!
        #Sometimes several samples have same complex and mutation but different experimental environment such as different ph
        
        #bad_complexes = [] #Flex cannot extract the pdb model files (wt and mt for each model) from the output of rosetta backrub for these complexes!
        #So it's better not to try and waste time on other samples that has the same complex but different mutations!
        
        samples_exist_output = os.listdir(self.flex_ddg_output_bu) #These are the samples that already exist in the output_bu
        #It is helpful to resume the process and continue with the rest of samples!
        for folder in samples_exist_output:
            protein_complex, mutate_complex, _ = folder.split('§')
            #current_sample = protein_complex + mutate_complex.split(',')[0][1:-1]
            current_sample = protein_complex + mutate_complex
            samples_already_processed.append(current_sample)
            
        reference_structures = os.listdir(self.raw_ref_struct)
        
        count = 0
        for index, row in dataframe.iterrows():
            if count > number:
                break
            try:
                protein_complex = str(row['#Pdb'])
                mutate_complex = str(row['Mutation(s)_cleaned'])
                interaction_region = str(row['iMutation_Location(s)'])
                complex_type = str(row['Hold_out_type'])
                experimental_method = str(row['Method'])
                b_affine_wt = str(row['Affinity_wt (M)'])
                b_affine_mt = str(row['Affinity_mut (M)'])
                
                if protein_complex not in reference_structures:
                    continue
                
                mutate_info = mutate_complex.split(',')
                #current_sample = protein_complex + mutate_info[0][1:-1]
                current_sample = protein_complex + mutate_complex
                folder_name = protein_complex + '§' + mutate_complex + '§' + experimental_method + "/"
                
                #if folder_name[:-1] not in self.reference_database:
                #    continue
                
                print("\n\nProcessing sample: index=" + str(index) + "\tComplex=" + protein_complex + '\tMutation=' + mutate_complex)
                
                if current_sample in samples_already_processed:
                    print("This sample is already processed!")
                    continue
                #if protein_complex in bad_complexes:
                #    print("This is a bad complex so let's not waste time on its samples and mutations!")
                #    continue
                
                samples_already_processed.append(current_sample)          
                            
                comp_info = protein_complex.split('_')
                pdb_code, chains1, chains2 = comp_info[0], comp_info[1], comp_info[2]
                
                #pdb_file = self.pdb_path + pdb_code + '.pdb'
                
                #if os.path.exists(self.flex_ddg_input):
                #    shutil.rmtree(self.flex_ddg_input)
                #os.makedirs(self.flex_ddg_input)
                
                if os.path.exists(self.flex_ddg_output):
                    shutil.rmtree(self.flex_ddg_output)
                os.makedirs(self.flex_ddg_output)
                            
                #in_path = self.flex_ddg_input + pdb_code + '/'
                #os.makedirs(in_path)
                #shutil.copyfile(pdb_file,in_path + pdb_code + '.pdb')
                
                out_path = self.flex_ddg_output + pdb_code + '/'
                
                #Load results of step 1
                shutil.copytree(os.path.join(self.raw_ref_struct, protein_complex), out_path)
                
                with open(out_path+"nataa_mutations.resfile", 'w') as fileH_mutations:
                    fileH_mutations.write("NATAA\n")
                    fileH_mutations.write("start\n")
                    for mutate in mutate_info:
                        original_aa = mutate[0]
                        mutated_ch = mutate[1]
                        position_aa = mutate[2:-1]
                        mutated_aa = mutate[-1]
                        fileH_mutations.write(position_aa + ' ' + mutated_ch + ' PIKAA ' + mutated_aa + "\n")
                
                
                print("\n\nSTEP 2\n\n")
            
                command = "python flex_ddg_run.py step2"
                os.system(command)
            
                #Extract structures created by local backrub sampling, local repacking, and global minimization processes
                command = "python flex_ddg_extract_structures.py " + self.flex_ddg_output
                os.system(command)
            
                #Check if pdb files are extracted successfully (for example for the first model)
                if not os.path.isfile(out_path+"/01/mut_01_"+str(backrub_steps).zfill(5)+".pdb"): #This problem must be resolved properly!
                    #bad_complexes.append(protein_complex)
                    raise Exception("The pdb files are not extracted!")
                
                
                #Save results of step 2
                print("\n\nSAVE: STEP2\n\n")
                #command = "cp -r " + out_path + " " + self.flex_ddg_output_bu + folder_name        
                #os.system(command)
                shutil.copytree(out_path, os.path.join(self.flex_ddg_output_bu, folder_name))
                with open(self.flex_ddg_output_bu + folder_name + "description", "w") as infile:
                    infile.write(protein_complex + "§" + mutate_complex + "§" + interaction_region + "§" + complex_type + "§" + experimental_method + "§" + b_affine_wt + "§" +  b_affine_mt)
                
            except Exception as e:
                print("\nOuch! An exception happened while processing sample: index=" + str(index) + "\tComplex=" + protein_complex + '\tMutation=' + mutate_complex + 
                      "\tException: " + str(e) + "\nMore information:\n" + traceback.format_exc())
                continue
            count+=1


    """
    Combination of LISA descriptions for interface (interaction signals) and Flex_ddG descriptions based on rosetta backrub for complex stability (energy terms)
    """
    def LisaFlexDescriptionAnalysis(self, name, models_folder_wt, models_folder_mt, number, backrub_steps, nstruct):
        
        results_flex_mt = pd.DataFrame(columns = self.inden_vector_skmp + self.score_vector_flex)
        results_flex_wt = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_flex)
        
        results_flex_total_mt = pd.DataFrame(columns = self.inden_vector_skmp + self.score_vector_flex)
        results_flex_total_wt = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_flex)
        
        results_lisa_mt = pd.DataFrame(columns = self.inden_vector_skmp + self.score_vector_lisa)
        results_lisa_wt = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_lisa)
        
        results_lisa_total_mt = pd.DataFrame(columns = self.inden_vector_skmp + self.score_vector_lisa)
        #results_lisa_total_br = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_lisa)
        results_lisa_total_wt = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_lisa)

        results_lisa_mt_original = pd.DataFrame(columns = self.inden_vector_skmp + self.score_vector_lisa_original)
        results_lisa_wt_original = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_lisa_original)
        
        results_lisa_total_mt_original = pd.DataFrame(columns = self.inden_vector_skmp + self.score_vector_lisa_original)
        #results_lisa_total_br_original = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_lisa_original)
        results_lisa_total_wt_original = pd.DataFrame(columns = self.inden_vector_skmp_wt + self.score_vector_lisa_original)
        
        b_affinities_expr_mt = []
        b_affinities_expr_wt = []
        b_affinities_expr_ddg = []
        b_affinities_pred_lisa_mt = []
        b_affinities_pred_lisa_wt = []
        b_affinities_pred_lisa_ddg = []
        b_affinities_pred_flex_mt = []
        b_affinities_pred_flex_wt = []
        b_affinities_pred_flex_ddg = []
    
        wt_already_processed = set()
        
    
        list_samples = os.listdir(models_folder_mt)
        #This sort is very important to help us to process all the samples of a specific complex consecutively 
        #because of the values of the binding affinity (predict and experiment) of wild-type see variable 'current_complex' 
        list_samples.sort() 
        
        for count, sample in enumerate(list_samples):
            if count > number:
                break
            
            try:
                
                if os.path.exists(self.flex_ddg_output):
                    shutil.rmtree(self.flex_ddg_output)
                os.makedirs(self.flex_ddg_output)
                
                if os.path.exists(self.flex_ddg_analysis):
                    shutil.rmtree(self.flex_ddg_analysis)
                os.makedirs(self.flex_ddg_analysis) 
                                        
                #Read the description of the sample
                with open(models_folder_mt + "/" + sample + "/" + "description") as infile:
                    sample_desc = infile.readline()
                
                protein_complex, mutate_complex, interaction_region, complex_type, experimental_method, b_affine_wt_tmp, b_affine_mt = sample_desc.split('§')
                
                process_wt = False
                if protein_complex not in wt_already_processed:
                    process_wt = True
                    
                    if os.path.exists(self.flex_ddg_output_raw):
                        shutil.rmtree(self.flex_ddg_output_raw)
                    os.makedirs(self.flex_ddg_output_raw)
                
                """
                This note is for the old version: 
                I observed that for wild-type structure of many complexes the experimental measurement of binding affinity is not consistent and it changes from sample to sample!
                So it won't be redundancy if we calculate the binding affinity and related features for wild-type structure of a complex repeatedly for its mutations!
                This issue must be discussed! But for now for each sample I process both wild-type and mutant! Later we can choose one feature vector for each complex!
                It might take more time because LISA processes the wild-type structure of a complex several times but it won't cause any problem because the results 
                of each sample is saved separately!
                """
                
                if process_wt:
                    b_affine_wt = self.Kd2DeltaG(float(b_affine_wt_tmp.replace("<","").replace(">","")))
                b_affine_mt = self.Kd2DeltaG(float(b_affine_mt.replace("<","").replace(">","")))
                
                iden_list = [protein_complex, mutate_complex, interaction_region, complex_type, experimental_method, b_affine_wt, b_affine_mt]
                iden_list_wt = [protein_complex, complex_type, experimental_method, b_affine_wt]
                iden_df = pd.DataFrame([iden_list], columns=self.inden_vector_skmp)
                iden_df_nst = pd.concat([iden_df]*nstruct, ignore_index=True)
                iden_df_wt = pd.DataFrame([iden_list_wt], columns=self.inden_vector_skmp_wt)
                iden_df_wt_nst = pd.concat([iden_df_wt]*nstruct, ignore_index=True)
                
                print("\n\nProcessing sample number: " + str(count) + "\tComplex=" + protein_complex + '\tMutation=' + mutate_complex)
                print(np.shape(b_affinities_expr_mt))
                print(np.shape(b_affinities_expr_ddg))
                print(np.shape(b_affinities_expr_wt))
                print(np.shape(b_affinities_pred_lisa_mt))
                print(np.shape(b_affinities_pred_lisa_ddg))
                print(np.shape(b_affinities_pred_lisa_wt))
                print(np.shape(b_affinities_pred_flex_mt))
                print(np.shape(b_affinities_pred_flex_ddg))
                print(np.shape(b_affinities_pred_flex_wt))
                
                comp_info = protein_complex.split('_')
                pdb_code, chains1, chains2 = comp_info[0], comp_info[1], comp_info[2]
                
                out_path = self.flex_ddg_output + pdb_code + '/'
                command = "cp -r " + models_folder_mt + "/" + sample + " " + out_path         
                os.system(command)
                os.remove(out_path + "description")
                os.remove(out_path + "chains_to_move.txt")
                os.remove(out_path + "interface_residues.resfile")
                os.remove(out_path + "nataa_mutations.resfile")
                
                if process_wt:
                    out_path_raw = self.flex_ddg_output_raw + pdb_code + '/'
                    command = "cp -r " + models_folder_wt + "/" + protein_complex + " " + out_path_raw         
                    os.system(command)
                    os.remove(out_path_raw + "chains_to_move.txt")
                    os.remove(out_path_raw + "interface_residues.resfile")
                
                command = "python analyze_flex_ddG.py " + self.flex_ddg_output
                os.system(command)
                
                #Load resulted scores
                ddg_flex_results = pd.read_csv(self.flex_ddg_analysis + '-results.csv', sep=',')
    
                #Refine the results and choose the right vector scores
                ddg_flex_results = ddg_flex_results.loc[(ddg_flex_results['score_function_name'] == 'fa_talaris2014') & (ddg_flex_results['backrub_steps'] == backrub_steps) & (ddg_flex_results['nstruct'] == nstruct)]
                ddg_flex_results = ddg_flex_results[self.score_vector_flex]
                #ddg_flex_results = ddg_flex_results.reset_index()
                #ddg_flex_results = ddg_flex_results.drop(ddg_flex_results.columns[0], axis=1)
                
                dg_flex_results_mt = pd.read_csv(self.flex_ddg_analysis + '-results_all_models_mt.csv', sep=',')
                dg_flex_results_mt = dg_flex_results_mt.loc[(dg_flex_results_mt['score_function_name'] == 'fa_talaris2014') & (dg_flex_results_mt['backrub_steps'] == backrub_steps) & (dg_flex_results_mt['nstruct'] == nstruct)] #This command is not necessary!
                dg_flex_results_mt = dg_flex_results_mt[self.score_vector_flex]
                
                if process_wt:
                    dg_flex_results_wt = pd.read_csv(self.flex_ddg_analysis + '-results_all_models_wt.csv', sep=',')
                    dg_flex_results_wt = dg_flex_results_wt.loc[(dg_flex_results_wt['score_function_name'] == 'fa_talaris2014') & (dg_flex_results_wt['backrub_steps'] == backrub_steps) & (dg_flex_results_wt['nstruct'] == nstruct)] #This command is not necessary!
                    dg_flex_results_wt = dg_flex_results_wt[self.score_vector_flex]
                
                
                results_lisa_tmp_mt = pd.DataFrame(columns = self.score_vector_lisa)
                results_lisa_tmp_mt_original = pd.DataFrame(columns = self.score_vector_lisa_original)
                
                if process_wt:
                    results_lisa_tmp_wt = pd.DataFrame(columns = self.score_vector_lisa)
                    results_lisa_tmp_wt_original = pd.DataFrame(columns = self.score_vector_lisa_original)
                    #results_lisa_tmp_br = pd.DataFrame(columns = self.score_vector_lisa)
                    #results_lisa_tmp_br_original = pd.DataFrame(columns = self.score_vector_lisa_original)
                    
                for i in range(len(os.listdir(out_path))):
                    pdb_file_mt = os.path.join(out_path, str(i+1).zfill(2), 'mut_'+str(i+1).zfill(2)+'_'+str(backrub_steps).zfill(5)+'.pdb')
                    lisa_dataframes_mt = self.AffinityPrediction(pdb_file_mt, chains1, chains2, True)
                    results_lisa_tmp_mt = pd.concat([results_lisa_tmp_mt, lisa_dataframes_mt[0]])
                    results_lisa_tmp_mt_original = pd.concat([results_lisa_tmp_mt_original, lisa_dataframes_mt[1]])
                    
                    if process_wt:
                        #pdb_file_br = os.path.join(out_path_raw, str(i+1).zfill(2), 'backrub_'+str(i+1).zfill(2)+'_'+str(backrub_steps).zfill(5)+'.pdb')
                        #lisa_dataframes_br = self.AffinityPrediction(pdb_file_br, chains1, chains2, True)
                        #results_lisa_tmp_br = pd.concat([results_lisa_tmp_br, lisa_dataframes_br[0]])
                        #results_lisa_tmp_br_original = pd.concat([results_lisa_tmp_br_original, lisa_dataframes_br[1]])
                    
                        pdb_file_wt = os.path.join(out_path_raw, str(i+1).zfill(2), 'wt_'+str(i+1).zfill(2)+'_'+str(backrub_steps).zfill(5)+'.pdb')
                        lisa_dataframes_wt = self.AffinityPrediction(pdb_file_wt, chains1, chains2, True)
                        results_lisa_tmp_wt = pd.concat([results_lisa_tmp_wt, lisa_dataframes_wt[0]])
                        results_lisa_tmp_wt_original = pd.concat([results_lisa_tmp_wt_original, lisa_dataframes_wt[1]])
                    
                
                results_lisa_tmp_mt = results_lisa_tmp_mt.reset_index(drop=True)
                results_lisa_total_mt = pd.concat([results_lisa_total_mt,
                                                   pd.concat([iden_df_nst, results_lisa_tmp_mt],axis=1)])
                
                results_lisa_tmp_mt_original = results_lisa_tmp_mt_original.reset_index(drop=True)
                results_lisa_total_mt_original = pd.concat([results_lisa_total_mt_original,
                                                            pd.concat([iden_df_nst, results_lisa_tmp_mt_original],axis=1)])
                
                results_lisa_mt = pd.concat([results_lisa_mt, pd.concat([iden_df, results_lisa_tmp_mt.mean().to_frame().transpose()],axis=1)])
                results_lisa_mt_original = pd.concat([results_lisa_mt_original, pd.concat([iden_df, results_lisa_tmp_mt_original.mean().to_frame().transpose()],axis=1)])
                
                if process_wt:
                    #results_lisa_tmp_br = results_lisa_tmp_br.reset_index(drop=True)
                    #results_lisa_total_br = pd.concat([results_lisa_total_br,
                    #                                   pd.concat([iden_df_wt_nst, results_lisa_tmp_br],axis=1)])
                    #results_lisa_tmp_br_original = results_lisa_tmp_br_original.reset_index(drop=True)
                    #results_lisa_total_br_original = pd.concat([results_lisa_total_br_original,
                    #                                            pd.concat([iden_df_wt_nst, results_lisa_tmp_br_original],axis=1)])
                    
                    results_lisa_tmp_wt = results_lisa_tmp_wt.reset_index(drop=True)
                    results_lisa_total_wt = pd.concat([results_lisa_total_wt,
                                                       pd.concat([iden_df_wt_nst, results_lisa_tmp_wt],axis=1)])
                    results_lisa_tmp_wt_original = results_lisa_tmp_wt_original.reset_index(drop=True)
                    results_lisa_total_wt_original = pd.concat([results_lisa_total_wt_original,
                                                                pd.concat([iden_df_wt_nst, results_lisa_tmp_wt_original],axis=1)])


                    results_lisa_wt = pd.concat([results_lisa_wt, pd.concat([iden_df_wt, results_lisa_tmp_wt.mean().to_frame().transpose()],axis=1)])
                    results_lisa_wt_original = pd.concat([results_lisa_wt_original, pd.concat([iden_df_wt, results_lisa_tmp_wt_original.mean().to_frame().transpose()],axis=1)])
                
                    results_flex_wt = pd.concat([results_flex_wt, 
                                                 pd.concat([iden_df_wt , ddg_flex_results.iloc[2].to_frame(name=0).T], axis=1)])
                                                 
                    results_flex_total_wt = pd.concat([results_flex_total_wt,
                                                       pd.concat([iden_df_wt_nst, dg_flex_results_wt],axis=1)])

                                                                        
                results_flex_mt = pd.concat([results_flex_mt, 
                                              pd.concat([iden_df , ddg_flex_results.iloc[1].to_frame(name=0).T], axis=1)])
            
                results_flex_total_mt = pd.concat([results_flex_total_mt,
                                                   pd.concat([iden_df_nst, dg_flex_results_mt],axis=1)])
            
                
                b_affine_lisa_mt = results_lisa_mt.iloc[0]['b_affine']
                b_affine_flex_mt = ddg_flex_results.iloc[1]["total_score"]
                
                if process_wt:
                    b_affine_lisa_wt = results_lisa_wt.iloc[0]['b_affine']
                    b_affine_flex_wt = ddg_flex_results.iloc[2]["total_score"]
                    
                b_affine_lisa_ddg = b_affine_lisa_mt - b_affine_lisa_wt
                """
                #Remove this note: 
                To avoid updating prediction of changes of binding affinity (because of updates in value of wildtype complex) each time for the same complex!
                (wildtype prediction remains constant for the complex as a reference)
                #b_affine_flex_ddg = float(ddg_flex_results.iloc[0]["total_score"])
                """
                b_affine_flex_ddg = b_affine_flex_mt - b_affine_flex_wt
            
                b_affinities_expr_mt.append((b_affine_mt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                b_affinities_expr_ddg.append((b_affine_mt-b_affine_wt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                b_affinities_pred_lisa_mt.append((b_affine_lisa_mt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                b_affinities_pred_lisa_ddg.append((b_affine_lisa_ddg,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                b_affinities_pred_flex_mt.append((b_affine_flex_mt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                b_affinities_pred_flex_ddg.append((b_affine_flex_ddg,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                
                if process_wt:
                    #Note: you should remove extra fields like interaction_region or mutate_complex from wt data
                    b_affinities_expr_wt.append((b_affine_wt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex)) 
                    b_affinities_pred_lisa_wt.append((b_affine_lisa_wt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                    b_affinities_pred_flex_wt.append((b_affine_flex_wt,interaction_region,complex_type,experimental_method,protein_complex,mutate_complex))
                
                wt_already_processed.add(protein_complex)
            except Exception as e:
                print("\nOuch! An exception happened while processing sample: Complex=" + protein_complex + '\tMutation=' + mutate_complex + 
                      "\tException: " + str(e) + "\nMore information:\n" + traceback.format_exc())
                continue
        
        source = "Backrub"
        output_files = self.res_path + source + '_' + name
        
        results_flex_wt.to_csv(output_files + '_results_flex_wt.csv', sep=';')
        results_flex_mt.to_csv(output_files + '_results_flex_mt.csv', sep=';')
        
        results_flex_total_wt.to_csv(output_files + '_results_flex_total_wt.csv', sep=';')
        results_flex_total_mt.to_csv(output_files + '_results_flex_total_mt.csv', sep=';')
        
        results_lisa_total_wt.to_csv(output_files + '_results_lisa_total_wt.csv', sep=';')
        results_lisa_total_mt.to_csv(output_files + '_results_lisa_total_mt.csv', sep=';')
        #results_lisa_total_br.to_csv(output_files + '_results_lisa_total_br.csv', sep=';')
        
        results_lisa_wt.to_csv(output_files + '_results_lisa_wt.csv', sep=';')
        results_lisa_mt.to_csv(output_files + '_results_lisa_mt.csv', sep=';')
        
        results_lisa_total_wt_original.to_csv(output_files + '_results_lisa_total_wt_original.csv', sep=';')
        results_lisa_total_mt_original.to_csv(output_files + '_results_lisa_total_mt_original.csv', sep=';')
        #results_lisa_total_br_original.to_csv(output_files + '_results_lisa_total_br_original.csv', sep=';')
        
        results_lisa_wt_original.to_csv(output_files + '_results_lisa_wt_original.csv', sep=';')
        results_lisa_mt_original.to_csv(output_files + '_results_lisa_mt_original.csv', sep=';')
        
        np.save(output_files+'_pred_flex_mt',b_affinities_pred_flex_mt)
        np.save(output_files+'_pred_flex_wt',b_affinities_pred_flex_wt)
        np.save(output_files+'_pred_flex_ddg',b_affinities_pred_flex_ddg)
        np.save(output_files+'_pred_lisa_mt',b_affinities_pred_lisa_mt)
        np.save(output_files+'_pred_lisa_wt',b_affinities_pred_lisa_wt)
        np.save(output_files+'_pred_lisa_ddg',b_affinities_pred_lisa_ddg)
        np.save(output_files+'_expr_mt',b_affinities_expr_mt)
        np.save(output_files+'_expr_wt',b_affinities_expr_wt)
        np.save(output_files+'_expr_ddg',b_affinities_expr_ddg)
        
obj1 = ConformationalModeling()

#obj1.ExploreConformationalSpace_Step1("ALL", 10000, 10000)
#obj1.ExploreConformationalSpace_Step2("ALL", 10000, 10000)
obj1.LisaFlexDescriptionAnalysis("ALL", "/home/mohseni/myThesis/GitLab/Backup_120320/Temp/raw_ref_struct", 
                                        "/home/mohseni/myThesis/GitLab/Backup_120320/Temp/output_bu", 7000000, 10000, 30)
#obj1.backrubFeatureAnalysis("ALL")
print("<<<DONE>>>")