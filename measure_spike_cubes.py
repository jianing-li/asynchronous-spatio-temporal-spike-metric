"""
Function: Measure distances of spike cubes.
Author: Jianing Li, lijianing@pku.edu.cn, Peking University, Feb. 16th, 2018.

"""
import numpy as np
from event_process import show_events
from event_process import spike_cube_processing
from spike_metric import spike_cube_metric
from event_process.show_events import  show_ON_OFF_events
import time


def spike_cubes_random_change_distances(events, temporal_steps, spatial_steps):
    """
    spike cubes distances in temporal and spatial random changing steps.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        temporal_steps    - the temporal random changing steps.
        spatial_steps    - the spatial random changing steps.

    Outputs:
    -------
        distances    - the distance of spike cubes after random changing based on 3d gaussian kernel method.

    """

    fft_l1_distances = np.zeros((len(temporal_steps), len(spatial_steps)))
    kernel_train_distances = np.zeros((len(temporal_steps), len(spatial_steps)))
    kernel_cube_distances = np.zeros((len(temporal_steps), len(spatial_steps)))

    for i, temporal_step in enumerate(temporal_steps):
        for j, spatial_step in enumerate(spatial_steps):

            strat_time = time.time()

            new_events = spike_cube_processing.spatial_temporal_random_change(events, temporal_step, spatial_step)


            # show_events.show_ON_OFF_events(new_events, width = 128, height =128, length = max(new_events[1, :]))


            # fft_l1_distances[i, j] = spike_cube_metric.fft_convolution_l1_norm(events, new_events, x_cube_size=32, y_cube_size=32,
            #                             t_cube_size=500, x_sigma=5, y_sigma=5, t_sigma=100) # fft convolution l1 norm measurement.
            #
            #
            # kernel_train_distances[i, j] = spike_cube_metric.kernel_method_spike_train(events, new_events,
            #                     x_cube_size=128, y_cube_size=128, t_cube_size=5000, sigma=5000) # kernel method spike train measurement.



            kernel_cube_distances[i, j] = spike_cube_metric.kernel_method_spike_cubes(events, new_events, width=128, height=128,
                                x_cube_size=128, y_cube_size=128, t_cube_size=5000, x_sigma=5, y_sigma=5, t_sigma=5000) # 3d gaussian kernel method.


            end_time = time.time()
            cost_time = end_time - strat_time

            print('cost_time = {}, distance = {}'.format(cost_time, kernel_cube_distances[i, j]))

    return fft_l1_distances, kernel_train_distances, kernel_cube_distances


def spike_cubes_removing_distances(events, removing_steps):
    """
    spike cubes distances for random removing spike numbers.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        removing_steps    - the removing spike numbers or steps.

    Outputs:
    -------
        distances    - the distance of spike cubes after random changing based on 3d gaussian kernel method.

    """

    fft_l1_distances = np.zeros((len(removing_steps)))
    kernel_train_distances = np.zeros((len(removing_steps)))
    kernel_cube_distances = np.zeros((len(removing_steps)))

    for i, removing_step in enumerate(removing_steps):

        new_events = spike_cube_processing.random_remove_spikes(events, int(removing_step)) # random removing spike numbers.


        # show_events.show_ON_OFF_events(new_events, width = 128, height =128, length = max(new_events[1, :]))


        strat_time = time.time()

        # fft_l1_distances[i] = spike_cube_metric.fft_convolution_l1_norm(events, new_events, x_cube_size=32, y_cube_size=32,
        #                             t_cube_size=500, x_sigma=5, y_sigma=5, t_sigma=100) # fft convolution l1 norm measurement.

        # kernel_train_distances[i] = spike_cube_metric.kernel_method_spike_train(events, new_events,
        #                     x_cube_size=128, y_cube_size=128, t_cube_size=5000, sigma=5000) # kernel method spike train measurement.


        kernel_cube_distances[i] = spike_cube_metric.kernel_method_spike_cubes(events, new_events, width=128, height=128,
                                                                      x_cube_size=128, y_cube_size=128,
                                                                      t_cube_size=5000, x_sigma=5, y_sigma=5,
                                                                      t_sigma=5000)  # 3d gaussian kernel method.

        end_time = time.time()
        cost_time = end_time - strat_time

        print('cost_time = {}, fft1_norm = {}, kernel_train = {}, kernel_cube = {}'.format(cost_time, fft_l1_distances[i], kernel_train_distances[i], kernel_cube_distances[i]))


    return fft_l1_distances, kernel_train_distances, kernel_cube_distances


def spike_cubes_displacement_distances(events, temporal_cyclic_displacement_steps, spatial_cyclic_displacement_steps):
    """
    spike cubes distances for cyclic displacement.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        cyclic_displacement_steps    - the removing spike numbers or steps.

    Outputs:
    -------
        distances    - the distance of spike cubes after random changing based on 3d gaussian kernel method.

    """

    fft_l1_distances = np.zeros((len(temporal_cyclic_displacement_steps), len(spatial_cyclic_displacement_steps)))
    kernel_train_distances = np.zeros((len(temporal_cyclic_displacement_steps), len(spatial_cyclic_displacement_steps)))
    kernel_cube_distances = np.zeros((len(temporal_cyclic_displacement_steps), len(spatial_cyclic_displacement_steps)))

    for i, temporal_cyclic_displacement_step in enumerate(temporal_cyclic_displacement_steps):
        for j, spatial_cyclic_displacement_step in enumerate(spatial_cyclic_displacement_steps):

            # new_events = spike_cube_processing.temporal_cyclic_displacement(events, temporal_cyclic_displacement_step) # cyclic displacement for spike cube.
            # new_events = spike_cube_processing.spatial_cyclic_displacement(events, 128, spatial_cyclic_displacement_step)  # cyclic displacement for spike cube.
            new_events = spike_cube_processing.temporal_spatial_cyclic_displacement(events, 128, temporal_cyclic_displacement_step, spatial_cyclic_displacement_step)

            strat_time = time.time()

            # show_events.show_ON_OFF_events(new_events, width = 128, height =128, length = max(new_events[1, :]))

            # fft_l1_distances[i, j] = spike_cube_metric.fft_convolution_l1_norm(events, new_events, x_cube_size=32, y_cube_size=32,
            #                             t_cube_size=500, x_sigma=5, y_sigma=5, t_sigma=100) # fft convolution l1 norm measurement.

            kernel_train_distances[i, j] = spike_cube_metric.kernel_method_spike_train(events, new_events,
                                x_cube_size=128, y_cube_size=128, t_cube_size=5000, sigma=5000) # kernel method spike train measurement.


            # kernel_cube_distances[i, j] = spike_cube_metric.kernel_method_spike_cubes(events, new_events, width=128, height=128,
            #                                                            x_cube_size=128, y_cube_size=128,
            #                                                            t_cube_size=5000, x_sigma=5, y_sigma=5,
            #                                                            t_sigma=5000)  # 3d gaussian kernel method.
            end_time = time.time()
            cost_time = end_time - strat_time

            print('cost_time = {}, average_events= {}, fft1_norm = {}, kernel_train = {}, kernel_cube = {}'.format(cost_time, events.shape[1]/(cost_time*1000), fft_l1_distances[i, j], kernel_train_distances[i, j], kernel_cube_distances[i, j]))


    return fft_l1_distances, kernel_train_distances, kernel_cube_distances


def polarity_percentage_chaning_distances(events, polarity_percentage_steps):
    """
    spike cubes distances for polarity percentage changing.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        polarity_percentage_steps    - the polarity random changing percentage steps.

    Outputs:
    -------
        distances    - the distance of spike cubes after random changing based on 3d gaussian kernel method.

    """

    fft_l1_distances = np.zeros((len(polarity_percentage_steps)))
    kernel_train_distances = np.zeros((len(polarity_percentage_steps)))
    kernel_cube_distances = np.zeros((len(polarity_percentage_steps)))

    for i, polarity_percentage_step in enumerate(polarity_percentage_steps):

        new_events = spike_cube_processing.events_polarity_random_changing(events, polarity_percentage_step)  # changing spike polarity.

        # show_events.show_ON_OFF_events(new_events, width = 128, height =128, length = max(new_events[1, :]))

        # fft_l1_distances[i] = spike_cube_metric.fft_convolution_l1_norm(events, new_events, x_cube_size=32, y_cube_size=32,
        #                             t_cube_size=500, x_sigma=5, y_sigma=5, t_sigma=100) # fft convolution l1 norm measurement.

        # kernel_train_distances[i] = spike_cube_metric.kernel_method_spike_train(events, new_events,
        #                     x_cube_size=128, y_cube_size=128, t_cube_size=5000, sigma=5000) # kernel method spike train measurement.

        # show_ON_OFF_events(new_events, width=128, height=128, length=max(events[1, :]))


        kernel_cube_distances[i] = spike_cube_metric.kernel_method_spike_cubes(events, new_events, width=128, height=128,
                                                                   x_cube_size=128, y_cube_size=128,
                                                                   t_cube_size=5000, x_sigma=5, y_sigma=5,
                                                                   t_sigma=5000)  # 3d gaussian kernel method.

        print('polarity_percentage_step={}, kernel_cube_distances={}'.format(polarity_percentage_step, kernel_cube_distances[i]))


    return fft_l1_distances, kernel_train_distances, kernel_cube_distances


def increasing_noise_distances(events, SNRs):
    """
    spike cubes distances for increasing random noise.

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        SNRs    - the ratio steps of signal to noise, namely spike numbers : noise numbers.

    Outputs:
    -------
        distances    - the distance of spike cubes after random changing based on 3d gaussian kernel method.

    """

    fft_l1_distances = np.zeros((len(SNRs)))
    kernel_train_distances = np.zeros((len(SNRs)))
    kernel_cube_distances = np.zeros((len(SNRs)))

    for i, SNR in enumerate(SNRs):

        new_events = spike_cube_processing.events_increasing_noises(events, SNR, 1, 128, 128)

        # fft_l1_distances[i] = spike_cube_metric.fft_convolution_l1_norm(events, new_events, x_cube_size=32, y_cube_size=32,
        #                             t_cube_size=500, x_sigma=5, y_sigma=5, t_sigma=100) # fft convolution l1 norm measurement.

        # kernel_train_distances[i] = spike_cube_metric.kernel_method_spike_train(events, new_events,
        #                     x_cube_size=128, y_cube_size=128, t_cube_size=5000, sigma=5000) # kernel method spike train measurement.

        kernel_cube_distances[i] = spike_cube_metric.kernel_method_spike_cubes(events, new_events, width=128,
                                                                               height=128,
                                                                               x_cube_size=128, y_cube_size=128,
                                                                               t_cube_size=5000, x_sigma=5, y_sigma=5,
                                                                               t_sigma=5000)  # 3d gaussian kernel method.

        print('SNR={}, kernel_cube_distances={}'.format(SNR, kernel_cube_distances[i]))


    return fft_l1_distances, kernel_train_distances, kernel_cube_distances






