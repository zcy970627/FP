#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import numpy as np
from numpy import asarray
from numpy.random import randn, rand, shuffle
import random
from hv_TUD import HyperVolume
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D

from datetime import datetime
import cProfile, pstats
# ACIP imports
import acip.utils.log as log
import acip.image as image
import acip.utils.file_io as file_io
import acip.config as config
import acip.optimization as opt
import acip.processing as proc

# Usage Example
# ./bin/processing/demo/demo_optimization.py -o path/to/results/file

# General parameters
SCRIPT_NAME = 'NorCAS2020_optimization'
repo_top_path = config.get_repo_top_dir()
test_data_path = config.get_test_data_dir()
max_bitwidth = 16

# Logging parameters
CONSOLE_LEVEL = log.VERBOSE
WRITE_LOGFILE = False # or none
LOGFILE_LEVEL = log.INFO
log_config = log.LoggingConfiguration(CONSOLE_LEVEL, WRITE_LOGFILE, LOGFILE_LEVEL)
log.setup(*log_config)
logger = logging.getLogger('acip.__main__')

PROFILE_RUN = False

# Script Parameters

# Simulation flag enable pooling of processor cores in simulation
simulation_flag = True

# Bitwidth of targeted pipeline
pipeline_bitwidth = 12

# Module Data Path
pre_lut_data_path  = test_data_path + '/module_parameters/processing/luts/lut1d_full/tonemap_iwlcs19_in_16_out_16.csv'
post_lut_data_path = test_data_path + '/module_parameters/processing/luts/lut1d_full/eotf_comp_iwlcs19_in_16_out_16.csv'
matrix_data_path = test_data_path + '/module_parameters/processing/color/matrices/alexa_wg_to_rec709.csv'
# Training Image Path
training_image_path = test_data_path + '/frames/synthetic/colorCube/colorCube16bit_noisySweep_16steps.tif'
# Results path
evaluation_results_path = repo_top_path + '/results/'

# NSGA2 options
init_pop_size=50    # initial population size
n_mu=50             # the number of individuals to select for the next generation
n_lambda=100        # the number of children to produce at each generation
p_cross=0.7         # crossover probability
p_mutation=0.3      # mutation probability
n_generations=1000  # number of generations
ga_options = opt.genetic.ga_setup.GaOptions(init_pop_size, n_mu, n_lambda, p_cross, p_mutation, n_generations)
fitness = ['Power', 'MaxDE', 'MeanDE']
n_fitness = len(fitness)

# Fixed reference point per fitness for hypervolume measurement, values preferable to higher than maximum of each fitness
hypervolume_reference = [1.18, 231.36, 84.85]
#hypervolume_reference = None

# Pipeline Module List
pipeline_module_list = ['uniSparseLut', 'colorMatrix', 'uniSparseLut']

# Module related parameters (power of 2)
# Uniform sections which is a power of 2, and uni_lut_max_sections <= 2^pipeline_bitwidth
uni_lut_max_sections = 32
# Log sections at the power of 2, and log_lut_max_sections <= pipeline_bitwidth
log_lut_max_sections = pipeline_bitwidth

## Helper Functions
hv = HyperVolume(hypervolume_reference)
# handles the parsing of output file name as input argument
def argparsing():
    parser = argparse.ArgumentParser(description="Lumalu NSGA-II Optimization")
    parser.add_argument("-o", "--output", nargs=1, type=str, required=False)
    args = parser.parse_args()
    return args

def get_output_filename(args):
    if args.output != None:
        out_file_name = args.output[0]
        logger.info("Output filename: {}".format(out_file_name))
    else:
        now = datetime.now()
        prefix = now.strftime('%Y-%m-%d_%H-%M-%S')
        image_name = os.path.basename(training_image_path).split('.')
        if (len(image_name)== 1):
            image_file = '' # no image file name for multiple image evaluations together
        else:
             # added image file name for single image evaluation results
            image_file = os.path.basename(training_image_path).split('.')[0]

        img_step = training_image_path[-12:-9]
        objectives = '_'
        for i in range(n_fitness):
            objectives += (fitness[i][0][0:2])

        out_file_name = 'NorCAS_opt_'+prefix+'_Step'+img_step+'_'+'Bitwidth'+str(pipeline_bitwidth)+ objectives+'_Gen'+str(n_generations) +'_Section'+str(uni_lut_max_sections)+'_nMu'+str(n_mu)+'_pCross'+str(p_cross)+'_limSum16.pickle'
        logger.info('No output filename defined. {} will be used instead.'.format(out_file_name))
    return out_file_name


    
def FindNeighbor(start_pt):
    
    pos1 = random.choice([1,2,3,9,10,11])
    if pos1 == 1:
        start_pt[0][1] = 1 - start_pt[0][1]
        
    if pos1 == 2:
        p1 = random.randint(0,5)
        start_pt[0][2] = 2 ** p1
        
        #regenerate position 3
        start_pt[0][3] = [0] * start_pt[0][2]
        for i in range(start_pt[0][2]):
            start_pt[0][3][i] = 2 ** random.randint(0,12)
            

        # lower bound
        while sum(start_pt[0][3]) < 16:
            b = random.randint(0,start_pt[0][2]-1)
            start_pt[0][3][b] = start_pt[0][3][b] * 2
            continue
            
        for m1 in range(start_pt[0][2]):
            while start_pt[0][3][m1] > (4096 // start_pt[0][2]):
                start_pt[0][3][m1] = start_pt[0][3][m1] // 2
                
    if pos1 == 3:
        while True:
            ind1 = random.randint(0,len(start_pt[0][3])-1)
            if start_pt[0][3][ind1] < (4096 // start_pt[0][2]) and sum(start_pt[0][3])-0.5*start_pt[0][3][ind1] >= 16:
                symbol1 = random.randint(0,1)
                if symbol1 == 0 and start_pt[0][3][ind1] != 1:
                    start_pt[0][3][ind1] = start_pt[0][3][ind1] // 2
                elif symbol1 == 1:
                    start_pt[0][3][ind1] = start_pt[0][3][ind1] * 2
                break
            elif start_pt[0][3][ind1] == (4096 // start_pt[0][2]):
                start_pt[0][3][ind1] = start_pt[0][3][ind1] // 2
                break
            elif sum(start_pt[0][3])-0.5*start_pt[0][3][ind1] < 16:
                start_pt[0][3][ind1] = start_pt[0][3][ind1] * 2
                break
                
    if pos1 == 9:
        start_pt[2][1] = 1 - start_pt[2][1]
        
    if pos1 == 10:
        p2 = random.randint(0,5)
        start_pt[2][2] = 2 ** p2
        
        #regenerate position 3
        start_pt[2][3] = [0] * start_pt[2][2]
        for i in range(start_pt[2][2]):
            start_pt[2][3][i] = 2 ** random.randint(0,12)
    
        while sum(start_pt[2][3]) < 16:
            b = random.randint(0,start_pt[2][2]-1)
            start_pt[2][3][b] = start_pt[2][3][b] * 2
            continue
            
        for m2 in range(start_pt[2][2]):
            while start_pt[2][3][m2] > (4096 // start_pt[2][2]):
                start_pt[2][3][m2] = start_pt[2][3][m2] // 2
                
    if pos1 == 11:
        while True:
            ind1 = random.randint(0,len(start_pt[2][3])-1)
            if start_pt[2][3][ind1] < (4096 // start_pt[2][2]) and sum(start_pt[2][3])-0.5*start_pt[2][3][ind1] >= 16:
                symbol2 = random.randint(0,1)
                if symbol2 == 0 and start_pt[2][3][ind1] != 1:
                    start_pt[2][3][ind1] = start_pt[2][3][ind1] // 2
                elif symbol2 == 1:
                    start_pt[2][3][ind1] = start_pt[2][3][ind1] * 2
                break
            elif start_pt[2][3][ind1] == (4096 // start_pt[2][2]):
                start_pt[2][3][ind1] = start_pt[2][3][ind1] // 2
                break
            elif sum(start_pt[2][3])-0.5*start_pt[2][3][ind1] < 16:
                start_pt[2][3][ind1] = start_pt[2][3][ind1] * 2
                break

    for q in range(2):        

        pos2 = random.choice([4,5,6,7,8])        
        if pos2 == 4:
            row = random.randint(0,2)
            col = random.randint(0,2)
            max_original = max(start_pt[1][row+1][1])
            start_pt[1][row+1][1][col] = random.randint(0,13)
            max_new = max(start_pt[1][row+1][1])

            if max_new != max_original:
                start_pt[1][row+1][2] = random.randint(0, max_new)
                start_pt[1][row+1][-1][0] = random.randint(0, 12 + start_pt[1][row+1][2])
                start_pt[1][row+1][-1][1] = random.randint(0, 12 + start_pt[1][row+1][2])


        if pos2 == 5:
            ind = random.randint(0,2)
            max_5 = max(start_pt[1][ind+1][1])
            start_pt[1][ind+1][2] = random.randint(0, max_5)
            start_pt[1][ind+1][-1][0] = random.randint(0, 12 + start_pt[1][ind+1][2])
            start_pt[1][ind+1][-1][1] = random.randint(0, 12 + start_pt[1][ind+1][2])


        if pos2 == 6:
            row = random.randint(0,2)
            col = random.randint(0,1)
            start_pt[1][row+1][3][col] = 3 - start_pt[1][row+1][3][col]

            #check position 7
            if start_pt[1][row+1][3][col] == 1 and start_pt[1][row+1][4][col] == 1:
                start_pt[1][row+1][4][col] = 0


        if pos2 == 7:
            #can only be changed when LSA
            a = []
            while True:
                for i in range(3):
                    for j in range(2):
                        a.append(start_pt[1][i+1][3][j])
                if sum(a) == 6:
                    row = random.randint(0,2)
                    col = random.randint(0,1)
                    start_pt[1][row+1][3][col] = 3 - start_pt[1][row+1][3][col]                
                    #check position 7
                    if start_pt[1][row+1][3][col] == 1 and start_pt[1][row+1][4][col] == 1:
                        start_pt[1][row+1][4][col] = 0

                row = random.randint(0,2)
                col = random.randint(0,1)
                if start_pt[1][row+1][3][col] == 2:
                    start_pt[1][row+1][4][col] = 1 - start_pt[1][row+1][4][col]
                    break
                else:
                    continue


            #check position 7
            if start_pt[1][row+1][3][col] == 1 and start_pt[1][row+1][4][col] == 1:
                start_pt[1][row+1][4][col] = 0

        if pos2 == 8:
            row = random.randint(0,2)
            col = random.randint(0,1)
            start_pt[1][row+1][-1][col] = random.randint(0,12 + start_pt[1][row+1][2])
        

    return start_pt



def hillclimbing(start_pt, n_iterations, fitness1, fitness2, fitness3, k):
    
    # initialization
    P=[]
    hv_list = []
    P.append([start_pt ,eval_fun(start_pt)])
    stagnation_counter = 0

    for i in range(1,n_iterations,1): #number of iterations
        logger.info("Generation{}".format(i))

        # get neighbour and its fitness
        candidate_x = FindNeighbor(start_pt) #find one neighbor for start_pt
        fitness_c_x = eval_fun(candidate_x) #get objective value of this neighbor
        candidate_x_fit = [candidate_x, fitness_c_x] #combine as an element in P
        y1, y2, y3 = fitness_c_x
        for j in range(len(P)): #make comparing with all elements in P
            fitness_j = P[j][1] #get fitness value in P

            if fitness_j[0]>=y1 and fitness_j[1]>=y2 and fitness_j[2]>=y3: # absolutely better
                stagnation_counter = 0
                P[j] = candidate_x_fit #replace
                start_pt = candidate_x

            elif y1<fitness_j[0] or y2<fitness_j[1] or y3<fitness_j[2]:  # pareto front
                P.append(candidate_x_fit)
                start_pt = candidate_x
                stagnation_counter = 0  
                break

            else: #absolutely worse
                if stagnation_counter >= k:
                    start_pt = RandomPick(P)
                    stagnation_counter = 0
                stagnation_counter += 1
        
        if i%100== 0:
            pareto_front = [P[i][-1] for i in range(len(P))]
            volume = hv.compute(pareto_front)
            print(volume)
            hv_list.append(volume)

    P1 = []
    [P1.append(i) for i in P if not i in P1]

    return P1, hv_list

def RandomPick(P):
    rand_idx = random.randint(0,len(P)-1)
    parent = P[rand_idx][0]
    return parent


## Main Script
if __name__ == "__main__":

    if PROFILE_RUN:
        pr = cProfile.Profile(builtins=False)
        pr.enable()

    # Setup Output filename
    args = argparsing()
    out_file_name = get_output_filename(args)

    # offset calculation w.r.to maximim bitdepth
    bitwidth_adjustment = max_bitwidth-pipeline_bitwidth
    stepsize = 2**(bitwidth_adjustment)

    # Read training image and create reference output
    if training_image_path.find('.tif'):
        img_in = image.io.tiffio.read(training_image_path).astype(np.int64) >> bitwidth_adjustment
    else:
        img_in = image.io.mpimgio.read(training_image_path).astype(np.int64) >> bitwidth_adjustment

    pre_lut_content  = np.array(file_io.read_from_csv(pre_lut_data_path, convert_to_int=True)[0::stepsize]) >> bitwidth_adjustment
    post_lut_content = np.array(file_io.read_from_csv(post_lut_data_path, convert_to_int=True)[0::stepsize]) >> bitwidth_adjustment

    ## Setup Evaluation - NOT IMPRORTANT
    ref_pipe = proc.applications.systems.lumalu.lumalu_pipelines.makeDefaultLumaluReferencePipeline(pipeline_bitwidth, pre_lut_content, matrix_data_path, post_lut_content)
    img_out_ref = ref_pipe.process_fast(img_in)
    fitness_options = opt.systems.lumalu.evaluation.make_default_lumalu_models(pipeline_bitwidth, fitness, img_in, img_out_ref)

    # Evaluation Functions
    def eval_fun(ind):
        test_pipe = opt.systems.lumalu.ind_to_pipe(ind, pipeline_bitwidth, [pre_lut_content, matrix_data_path, post_lut_content])
        return opt.systems.lumalu.fitness_evaluation(test_pipe, fitness_options, fitness)


    lumalu_genetics = opt.genetic.operations.genetic_operations.ModularOperations(pipeline_module_list, pipeline_bitwidth, uni_lut_max_sections, log_lut_max_sections)

    n_iterations = 40000
    k = 5 # stagnation parameter
    hv_all = []

    for i in range(5):
        start_pt = lumalu_genetics.randomInd()
        fitness1, fitness2, fitness3 = eval_fun(start_pt) #objective   
        P, hv_list = hillclimbing(start_pt, n_iterations, fitness1, fitness2, fitness3, k)
        
        logger.info("Best configuration {}".format(P))
        logger.info("Length of P {}".format(len(P)))

        pareto_front = [P[i][-1] for i in range(len(P))]
        hv = HyperVolume(hypervolume_reference)
        front = pareto_front
        volume = hv.compute(front)
        hv_all.append(volume)
        
        fig=plt.figure(figsize=(16,16))
        ax=fig.add_subplot(projection='3d')
        fitness_P = [pareto_front[i] for i in range(len(pareto_front))]
        x, y, z = np.array(fitness_P).T
        ax.scatter(x, y, z)
        ax.set_zlabel('MeanDE')
        ax.set_ylabel('MaxDE')
        ax.set_xlabel('Power')
        plt.savefig('/work/publications/norcas2020/experiments/results/3/3dPareto'+str(i+1)+'.png')
        plt.show()

        for k in range(0, len(pareto_front)):
            plt.plot(pareto_front[k][0],pareto_front[k][1],"+r")
        plt.savefig('/work/publications/norcas2020/experiments/results/3/2dPareto'+str(i+1)+'.png')
        plt.show()

        f = open('/work/publications/norcas2020/experiments/results/3/hv_list'+str(i+1)+'.pkl','wb')
        pickle.dump(hv_list,f)
        f.close()

        f = open('/work/publications/norcas2020/experiments/results/3/P'+str(i+1)+'.pkl','wb')
        pickle.dump(P,f)
        f.close()

        f = open('/work/publications/norcas2020/experiments/results/3/ParetoFront'+str(i+1)+'.pkl','wb')
        pickle.dump(pareto_front,f)
        f.close()

        x = list(range(len(hv_list)))
        x_axis = [k * 100 for k in x]
        plt.plot(x_axis, hv_list)
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume')
        plt.savefig('/work/publications/norcas2020/experiments/results/3/Convergence'+str(i+1)+'.png')
        plt.show()

    file = open('/work/publications/norcas2020/experiments/results/3/HV3.txt','w')
    for hv in hv_all:
        file.write(str(hv))
        file.write('\n')
    file.write('Average Hypervolume of Neighbor Function 3 after 40000 Generations is')
    file.write('\b')
    file.write(str(np.mean(hv_all)))
    file.close()

    if PROFILE_RUN:
        pr.disable()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats()



