#!/usr/bin/env python3
import argparse
import logging
import os
import csv
import copy
import time
import cProfile, pstats
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import acip.utils.log as log
import acip.utils.file_io as file_io
import acip.image as image
import acip.config as config
import acip.processing as proc
import acip.image as image
import acip.optimization as opt


# General parameters
SCRIPT_NAME = 'IJNC Lumalu baseline solutions'
repo_top_path = config.get_repo_top_dir()
test_data_path = config.get_test_data_dir()
# Logging parameters
CONSOLE_LEVEL = log.INFO
WRITE_LOGFILE = False # or none
LOGFILE_LEVEL = log.INFO
log_config = log.LoggingConfiguration(CONSOLE_LEVEL, WRITE_LOGFILE, LOGFILE_LEVEL)
log.setup(*log_config)
logger = logging.getLogger('acip.__main__')

# Paths
prelut_data_path  = test_data_path + '/module_parameters/processing/luts/lut1d_full/tonemap_iwlcs19_in_16_out_16.csv'
postlut_data_path = test_data_path + '/module_parameters/processing/luts/lut1d_full/eotf_comp_iwlcs19_in_16_out_16.csv'
matrix_data_path = test_data_path + '/module_parameters/processing/color/matrices/alexa_wg_to_rec709.csv'
# Training Image Path
training_image_path = test_data_path + '/frames/synthetic/colorCube/colorCube16bit_noisySweep_128steps.tif'
# Output path
evaluation_results_path = repo_top_path + '/results/publications/norcas2020/csv_files/'
csv_file_name = 'pareto_display_rendering_baseline_points.csv'

REF_BITWIDTH = 12
ga_fitness = ['Power', 'MaxDE', 'MeanDE']
smallest_base = 6

write_csv = True

is_csv_dir = os.path.isdir(evaluation_results_path)
if (is_csv_dir == False) and write_csv:
    os.makedirs(evaluation_results_path)

def argparsing():
    parser = argparse.ArgumentParser(description="Reference Metric Calculation Tool")
    parser.add_argument("-i", "--input", nargs=1, type=str, required=True, help="Path to image (relative to repo top dir)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    ref_bitwidth = REF_BITWIDTH
    bitwidth_adjustment_for_ref = 16-ref_bitwidth
    stepsize = 2**(bitwidth_adjustment_for_ref)

    im_train = image.io.tiffio.read(training_image_path).astype(int) >> bitwidth_adjustment_for_ref

    prelut_content  = np.array(file_io.read_from_csv(prelut_data_path, convert_to_int=True)[0::stepsize]) >> bitwidth_adjustment_for_ref
    postlut_content = np.array(file_io.read_from_csv(postlut_data_path, convert_to_int=True)[0::stepsize]) >> bitwidth_adjustment_for_ref
    ref_pipe = proc.applications.systems.lumalu.lumalu_pipelines.makeDefaultLumaluReferencePipeline(ref_bitwidth, prelut_content, matrix_data_path, postlut_content)
    im_train_ref = ref_pipe.process_fast(im_train)

    fitness_options = opt.systems.camera.evaluation.make_default_camera_models(ref_bitwidth, ga_fitness, im_train, im_train_ref, sim_flag=True)


    # Compare bit-reduced stuff
    bitwidth_list = []
    area_list  = []
    power_list = []
    mean_de_list = []
    max_de_list  = []

    for test_bitwidth in range(smallest_base, ref_bitwidth+1):
        print(test_bitwidth)
        bitwidth_list.append(test_bitwidth)

        bitwidth_adjustment_for_test = ref_bitwidth-test_bitwidth
        stepsize = 2**(bitwidth_adjustment_for_test)

        prelut_content_adj  = prelut_content[0::stepsize] >> bitwidth_adjustment_for_test
        postlut_content_adj = postlut_content[0::stepsize] >> bitwidth_adjustment_for_test
        im_train_for_test = im_train >> bitwidth_adjustment_for_test

        test_pipe   = proc.applications.systems.lumalu.lumalu_pipelines.makeDefaultLumaluReferencePipeline(test_bitwidth, prelut_content_adj, matrix_data_path, postlut_content_adj)
        im_test_out = test_pipe.process_fast(im_train_for_test).astype(int) << bitwidth_adjustment_for_test
        quality = opt.models.quality.QualityEstimate.from_test_image(im_test_out, fitness_options.quality_options)

        # Test results
        area = test_pipe.report_area(fitness_options.area_options)
        area.print()
        power = test_pipe.report_power(fitness_options.power_options)
        area_list.append(area.combined())
        power_list.append(power.combined())
        mean_de_list.append(quality.data.mean_delta_e)
        max_de_list.append(quality.data.max_delta_e)


    for i in range(0,len(bitwidth_list)):
        logger.info('Bitwidth: {:2d} | Area: {:1.4f} | Power: {:5.2f} | MeanDE: {:02.4f} | MaxDE: {:02.4f}'.format(bitwidth_list[i], area_list[i], power_list[i]*1000, mean_de_list[i], max_de_list[i]))

    if write_csv:
        with open(evaluation_results_path+csv_file_name, 'w') as f:
            writer = csv.writer(f, quotechar = "'")
            # Headline
            writer.writerow(['power', 'maxDeltaE', 'label'])
            for i in range(len(bitwidth_list)-1):
                if i < 10:
                    row = [power_list[i], max_de_list[i], '\"B' + str(bitwidth_list[i]) + ' \"']
                else:
                    row = [power_list[i], max_de_list[i], '\"B' + str(bitwidth_list[i]) + '\"']
                writer.writerow(row)
            row = [power_list[-1], max_de_list[-1], '\"Ref\"']
            writer.writerow(row)
        logger.info("CSV file generated successfully: {} ...!!!".format(evaluation_results_path + csv_file_name))
