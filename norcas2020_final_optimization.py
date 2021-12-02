#!/usr/bin/env python3
import argparse
import logging
import os
import numpy as np

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
evaluation_results_path = repo_top_path + '/results/optimization/nsga_two/lumalu/'

# NSGA2 options
init_pop_size=50    # initial population size
n_mu=50             # the number of individuals to select for the next generation
n_lambda=100        # the number of children to produce at each generation
p_cross=0.7         # crossover probability
p_mutation=0.3      # mutation probability
n_generations=400  # number of generations
ga_options = opt.genetic.ga_setup.GaOptions(init_pop_size, n_mu, n_lambda, p_cross, p_mutation, n_generations)
ga_fitness = ['Power', 'MaxDE', 'MeanDE']
n_fitness = len(ga_fitness)

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
            objectives += (ga_fitness[i][0][0:2])

        out_file_name = 'NorCAS_opt_'+prefix+'_Step'+img_step+'_'+'Bitwidth'+str(pipeline_bitwidth)+ objectives+'_Gen'+str(n_generations) +'_Section'+str(uni_lut_max_sections)+'_nMu'+str(n_mu)+'_pCross'+str(p_cross)+'_limSum16.pickle'
        logger.info('No output filename defined. {} will be used instead.'.format(out_file_name))
    return out_file_name


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
        img_in = image.io.tiffio.read(training_image_path).astype(np.int) >> bitwidth_adjustment
    else:
        img_in = image.io.mpimgio.read(training_image_path).astype(np.int) >> bitwidth_adjustment

    pre_lut_content  = np.array(file_io.read_from_csv(pre_lut_data_path, convert_to_int=True)[0::stepsize]) >> bitwidth_adjustment
    post_lut_content = np.array(file_io.read_from_csv(post_lut_data_path, convert_to_int=True)[0::stepsize]) >> bitwidth_adjustment

    # Setup Evaluation
    ref_pipe = proc.applications.systems.lumalu.lumalu_pipelines.makeDefaultLumaluReferencePipeline(pipeline_bitwidth, pre_lut_content, matrix_data_path, post_lut_content)
    img_out_ref = ref_pipe.process_fast(img_in)
    fitness_options = opt.systems.lumalu.evaluation.make_default_lumalu_models(pipeline_bitwidth, ga_fitness, img_in, img_out_ref)

    # Evaluation Functions
    def eval_fun(ind):
        test_pipe = opt.systems.lumalu.ind_to_pipe(ind, pipeline_bitwidth, [pre_lut_content, matrix_data_path, post_lut_content])
        return opt.systems.lumalu.fitness_evaluation(test_pipe, fitness_options, ga_fitness)

    # Setup Optimization
    lumalu_genetics = opt.genetic.operations.genetic_operations.ModularOperations(pipeline_module_list, pipeline_bitwidth, uni_lut_max_sections, log_lut_max_sections)
    toolbox, stats, hof = opt.genetic.ga_setup.setup(lumalu_genetics, eval_fun, n_fitness, simulation_flag)

    # Run Optimization
    pop, hof, logbook, hyp_stats = opt.genetic.ga_process.run(ga_options, toolbox, stats, hof, hypervolume_reference)
    print(hyp_stats)
    # Save optimization results to the pickle file
    opt.systems.lumalu.file_io.save_data(evaluation_results_path, out_file_name, pre_lut_data_path, matrix_data_path, post_lut_data_path, pipeline_module_list, pop, hof, n_generations, hyp_stats, training_image_path)

    if PROFILE_RUN:
        pr.disable()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats()
