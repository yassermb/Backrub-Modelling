#!/usr/bin/python

import socket
import sys
import os
import shutil
import subprocess

use_multiprocessing = True
if use_multiprocessing:
    import multiprocessing
    max_cpus = 35 #2 # We might want to not run on the full number of cores, as Rosetta take about 2 Gb of memory per instance

rosetta_scripts_path = os.path.expanduser("~/myStage/Softwares/Rosetta/rosetta_src_3.10_bundle/rosetta_src_2018.33.60351_bundle/main/source/bin/rosetta_scripts.linuxgccrelease")
nstruct = 30 # Normally 35
#max_minimization_iter = 5 # Normally 5000
max_minimization_iter = 5000 # Normally 5000
#abs_score_convergence_thresh = 200.0 # Normally 1.0
abs_score_convergence_thresh = 1.0 # Normally 1.0
#number_backrub_trials = 10 # Normally 35000
#number_backrub_trials = 35000 # Normally 35000
number_backrub_trials = 10000 # Normally 35000
backrub_trajectory_stride = 10000 #5 or 7000. Can be whatever you want, if you would like to see results from earlier time points in the backrub trajectory. 7000 is a reasonable number, to give you three checkpouints for a 35000 step run, but you could also set it to 35000 for quickest run time (as the final minimization and packing steps will only need to be run one time).
path_to_script_step1 = 'ddG-backrub_Step1.xml'
path_to_script_step2 = 'ddG-backrub_Step2.xml'

def run_flex_ddg_step1( name, input_path, input_pdb_path, chains_to_move, nstruct_i ):
    output_directory = os.path.join('Temp/output', os.path.join( name, '%02d' % nstruct_i ) )
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    flex_ddg_args = [
        os.path.abspath(rosetta_scripts_path),
        "-s %s" % os.path.abspath(input_pdb_path),
        '-parser:protocol', os.path.abspath(path_to_script_step1),
        '-parser:script_vars',
        'chainstomove=' + chains_to_move,
        'inter_resfile_relpath=' + os.path.abspath( os.path.join( input_path, 'interface_residues.resfile' ) ),
        'number_backrub_trials=%d' % number_backrub_trials,
        'max_minimization_iter=%d' % max_minimization_iter,
        'abs_score_convergence_thresh=%.1f' % abs_score_convergence_thresh,
        'backrub_trajectory_stride=%d' % backrub_trajectory_stride ,
        '-restore_talaris_behavior',
        '-in:file:fullatom',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy false',
        '-ex1',
        '-ex2',
    ]

    log_path = os.path.join(output_directory, 'rosetta.out')

    print('Running Rosetta with args:')
    print(' '.join(flex_ddg_args))
    print('Output logged to:', os.path.abspath(log_path))
    print

    outfile = open(log_path, 'w')
    process = subprocess.Popen(flex_ddg_args, stdout=outfile, stderr=subprocess.STDOUT, close_fds = True, cwd = output_directory)
    returncode = process.wait()
    outfile.close()
    
    #shutil.copyfile(os.path.join( input_path, 'chains_to_move.txt' ),os.path.join( 'Temp/output', name, 'chains_to_move.txt' ))
    #shutil.copyfile(os.path.join( input_path, 'nataa_mutations.resfile' ),os.path.join( 'Temp/output', name, 'nataa_mutations.resfile' ))
    #shutil.copyfile(os.path.join( input_path, 'interface_residues.resfile' ),os.path.join( 'Temp/output', name, 'interface_residues.resfile' ))
    #os.remove(os.path.join( 'Temp/output', name, str(nstruct_i).zfill(2), 'rosetta.out' ))
    #os.remove(os.path.join( 'Temp/output', name, str(nstruct_i).zfill(2), 'ddG.db3' ))
    #os.remove(os.path.join( 'Temp/output', name, str(nstruct_i).zfill(2), 'score.sc' ))
    #os.remove(os.path.join( 'Temp/output', name, str(nstruct_i).zfill(2), 'struct.db3' ))
    
    
def run_flex_ddg_step2( name, input_path, input_pdb_path, chains_to_move, nstruct_i ):
    output_directory = os.path.join('Temp/output', os.path.join( name, nstruct_i ) )
    #if not os.path.isdir(output_directory):
    #    os.makedirs(output_directory)
    flex_ddg_args = [
        os.path.abspath(rosetta_scripts_path),
        "-s %s" % os.path.abspath(input_pdb_path),
        '-parser:protocol', os.path.abspath(path_to_script_step2),
        '-parser:script_vars',
        'chainstomove=' + chains_to_move,
        'inter_resfile_relpath=' + os.path.abspath( os.path.join( input_path, 'interface_residues.resfile' ) ),
        'mutate_resfile_relpath=' + os.path.abspath( os.path.join( input_path, 'nataa_mutations.resfile' ) ),
        'br_pose_relpath=' + os.path.abspath( os.path.join( input_path, nstruct_i, "backrub_"+nstruct_i+"_"+str(number_backrub_trials).zfill(5)+".pdb" ) ),
        'number_backrub_trials=%d' % number_backrub_trials,
        'max_minimization_iter=%d' % max_minimization_iter,
        'abs_score_convergence_thresh=%.1f' % abs_score_convergence_thresh,
        'backrub_trajectory_stride=%d' % backrub_trajectory_stride ,
        '-restore_talaris_behavior',
        '-in:file:fullatom',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy false',
        '-ex1',
        '-ex2',
    ]

    log_path = os.path.join(output_directory, 'rosetta.out')

    print('Running Rosetta with args:')
    print(' '.join(flex_ddg_args))
    print('Output logged to:', os.path.abspath(log_path))
    print

    outfile = open(log_path, 'w')
    process = subprocess.Popen(flex_ddg_args, stdout=outfile, stderr=subprocess.STDOUT, close_fds = True, cwd = output_directory)
    returncode = process.wait()
    outfile.close()


if __name__ == '__main__':

    step_function = run_flex_ddg_step1
    if sys.argv[1] == "step1":
        cases = []
        for nstruct_i in range(1, nstruct + 1 ):
            for case_name in os.listdir('Temp/inputs'):
                case_path = os.path.join('Temp/inputs', case_name )
                for f in os.listdir(case_path):
                    if f.endswith('.pdb'):
                        input_pdb_path = os.path.join( case_path, f )
                        break
    
                with open( os.path.join( case_path, 'chains_to_move.txt' ), 'r' ) as f:
                    chains_to_move = f.readlines()[0].strip()
    
                cases.append( (case_name, case_path, input_pdb_path, chains_to_move, nstruct_i) )

    elif sys.argv[1] == "step2":
        step_function = run_flex_ddg_step2
        cases = []
        case_name = os.listdir('Temp/output')[0]
        case_path = os.path.join('Temp/output', case_name )
        for nstruct_i in os.listdir(case_path):
            if not os.path.isdir(os.path.join(case_path, nstruct_i)):
                continue
            input_pdb_path = os.path.join(case_path,nstruct_i.zfill(2),"wt_"+nstruct_i.zfill(2)+"_"+str(number_backrub_trials).zfill(5)+".pdb")
            print(input_pdb_path)
            with open( os.path.join( case_path, 'chains_to_move.txt' ), 'r' ) as f:
                    chains_to_move = f.readlines()[0].strip()
            cases.append( (case_name, case_path, input_pdb_path, chains_to_move, nstruct_i.zfill(2)) )
        
    if use_multiprocessing:
        pool = multiprocessing.Pool( processes = min(max_cpus, multiprocessing.cpu_count()) )

    for args in cases:
        if use_multiprocessing:
            pool.apply_async( step_function, args = args )
        else:
            step_function( *args )

    if use_multiprocessing:
        pool.close()
        pool.join()
