# given fixations, outputs ave entropy value for each fixation

import cv2
import time
import csv
from numpy import genfromtxt
import numpy as np
import pickle
import pandas as pd
import os

"""
######## funcs in this file ########
get_ave_entropy: 
    - gets the average normalized entropy for each frame in video (for full video)
reorg_fix
    - given a file name, outputs a list of list of dicts for input into get_entropy_values
get_entropy_values:
    - gets ave entropy value for each fixation (for a subject)
write_fix_vid:
    - given fixations and condition, outputs a video of fixations layered on (entropy) video 
resize_reg_vid:
    - given a video, outputs a the same video but resized (for input into double view)
write_fix_vid_double:
    - like write_fix_vid, but stacks the normal + entropy view on top of each other
"""

# os.chdir('/Users/sydney/Documents/Research/SCCN/street_view_vids')

def get_ave_entropy(which_video):
    """
    gets the average normalized entropy for each frame for first 1000 frames
    test with: ave_entropy, timelapsed = get_ave_entropy(video_name)
    :param which_video: bool, 0 if night, 1 if day
    :return ave_entropy, timelapsed: np array of ave entropy values, time taken to run
    """
    start = time.time()
    # get video
    if which_video:
        video = cv2.VideoCapture('DOWNTOWN DAY-entropy.mp4')
        max_vals = genfromtxt('downtown_day_max_entropy.csv', delimiter=',')
    else:
        video = cv2.VideoCapture('DOWNTOWN NIGHT-entropy.mp4')
        max_vals = genfromtxt('downtown_night_max_entropy.csv', delimiter=',')

    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # init entropy array
    ave_entropy = np.zeros(frame_count)

    for i in range(0, frame_count):
        ret, frame = video.read()
        frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)
        big_val = 0
        for x in range(0, int(h_video)):
            for y in range(0, int(w_video)):
                pixel = frame[x, y]
                big_val = big_val + sum(pixel)

        ave_entropy[i] = (big_val / (w_video * h_video)) / 765 * max_vals[int(frame_num)-1]
        # ave_entropy.append((big_val/(w_video*h_video))/765)

        if i % 1000 == 0:
            print('Completed ' + str(i) + ' frames')
            pause = time.time()
            print('Time elapsed: ' + str(pause-start))

    stop = time.time()
    timelapsed = stop - start

    return ave_entropy, timelapsed


def reorg_fix(file):
    """
    given a file name, outputs a list of list of dicts for input into get_entropy_values
    :param file: str, name of file exported from MATLAB
                channel labels:
                    - isFixation
                    - Original time
                    - New time (based on data collection srate)
                    - Frame count (which frame, aligns with original time)
                    - GIP x coordinate
                    - GIP y coordinate
    :return: master: list of list of dicts with time stamps (old and fixed), frame num, xycords for each fixation
             timelapsed: time taken to run
    """
    start = time.time()
    # import file
    fix = genfromtxt(file, delimiter=',')
    # init master list
    master = []
    # init list for each fixation
    fixation = []
    # for each col in struct:
    for i in range(0, fix.shape[1]):
        # if bool is true:
        if fix[0, i]:  # check the bool val for that col
            # write values into dict
            fix_point = {'og_time': fix[1, i], 'time': fix[2, i], 'frame': fix[3, i], 'xcord': fix[4, i],
                         'ycord': fix[5, i]}
            # check if next value is 1 or 0
            fixation.append(fix_point)
        else:
            if fixation:  # if there are items in fixation
                # add fixation to big list
                master.append(fixation.copy())
                fixation = []
    # for last fixation
    if fixation:
        master.append(fixation.copy())

    stop = time.time()
    timelapsed = stop - start

    return master, timelapsed


def get_entropy_values(which_video, fixations):
    """
    gets ave entropy value for each fixation
    test with
    :param which_video: bool, 0 is night, 1 is day
    :param fixations: list, generated from reorg_fix
    :return entropy, timelapsed: pandas df with important info, time taken to get it
    """
    start = time.time()

    # import video
    if which_video:
        video = cv2.VideoCapture('DOWNTOWN DAY-entropy.mp4')
        max_vals = genfromtxt('downtown_day_max_entropy.csv', delimiter=',')
    else:
        video = cv2.VideoCapture('DOWNTOWN NIGHT-entropy.mp4')
        max_vals = genfromtxt('downtown_night_max_entropy.csv', delimiter=',')

    # get width and height of video
    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # init pandas df for entropy vals
    cols = ['Ave_Entropy', 'Start_Time', 'End_Time', 'Duration']  # TODO: add frame start and end?
    entropy = pd.DataFrame(columns=cols)

    for each_fix, index in zip(fixations, np.arange(0, len(fixations))):
        # get important values
        start_time = each_fix[0].get('time')  # note: using 'time' and not 'og_time' for more accuracy
        end_time = each_fix[-1].get('time')
        duration = end_time - start_time
        # create empty list for entropy vals to average
        entropy_vals = []
        for fix in each_fix:
            # get frame, set video to that frame
            frame_num = fix.get('frame')
            video.set(1, frame_num)
            # read frame
            ret, frame = video.read()
            # find pixel associated with fixation
            x_coord = int(np.around((fix.get('xcord') * h_video), decimals=0))
            y_coord = int(np.around((1 - fix.get('ycord')) * w_video, decimals=0))
            # get pixel values
            try:  # this is here for debugging TODO: can probably remove now
                pixel = frame[x_coord, y_coord]
            except IndexError:
                print('Index: ' + str(index))
                print('fix: ')
                print(fix)
                print('Coords: ' + str(x_coord) + ',' + str(y_coord))
            except:
                print('other error :(')
            # get entropy value
            ent_val = np.mean(pixel) / 256  # TODO: or, do sum, and later divide?
            # multiply by max to get un-normed value
            ent_val = ent_val * max_vals[int(frame_num)]
            # input entropy value into df
            entropy_vals.append(ent_val)

        # insert value to entropy array
        ave_entropy = np.mean(entropy_vals)
        fix_row = pd.Series([ave_entropy, start_time, end_time, duration], index=entropy.columns)

        entropy = entropy.append(fix_row, ignore_index=True)

    video.release()
    stop = time.time()
    timelapsed = stop - start

    return entropy, timelapsed


def write_fix_vid(file, which_video):
    """
    given fixations and condition, outputs a video of fixations layered on (entropy) video
    :param file: str, file name of fixations  TODO: don't really need this, actually
    :param which_video: bool, 1 if day, 0 if night
    :return: Null
    """
    # get original video
    if which_video:
        video = cv2.VideoCapture('DOWNTOWN DAY-entropy.mp4')
    else:
        video = cv2.VideoCapture('DOWNTOWN NIGHT-entropy.mp4')

    # get width and height
    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # get fixations
    fix = genfromtxt(file, delimiter=',')

    # init videowriter obj
    out_name = 'pilot03_day_fix.avi'  # TODO: change this
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (int(w_video), int(h_video))
    out = cv2.VideoWriter(out_name, fourcc, 90, frame_size)

    # set color
    color = (255, 0, 0)
    # loop through while writing frames
    for i in range(0, 1000):  # first 1000 frames for testing # range(0, fix.shape[1]):
        # set some parameters
        isFix = fix[0, i]
        frame_num = int(fix[3, i])
        xcord = fix[4, i]
        ycord = fix[5, i]
        # get that specific frame
        video.set(1, frame_num)
        ret, frame = video.read()

        if isFix:  # if is fixation, draw fixation
            # get coordinates
            x = int(xcord * w_video)
            y = int(ycord * h_video)
            # draw crosshairs
            cv2.drawMarker(frame, (x, y), color, markerType=0, thickness=5)

        # write to outfile
        out.write(frame)

    video.release()
    out.release()

    print('Finished!')


def resize_reg_vid(which_video, proportion):
    """
    given a video, outputs a the same video but resized (for input into double view)
    :param which_video: bool, 1 if day, 0 if night
    :param proportion: value between 0-1; percentage to decrease/increase by (should be 0.5)
    :return: Null
    """
    # import orig vid + give name to new vid
    if which_video:
        video = cv2.VideoCapture('DOWNTOWN DAY.mp4')
        out_name = 'DOWNTOWN_DAY_resized.avi'
    else:
        video = cv2.VideoCapture('DOWNTOWN NIGHT.mp4')
        out_name = 'DOWNTOWN_NIGHT_resized.avi'

    # get important parameters
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # calculate new dimensions
    dims = (int(w_video * proportion), int(h_video * proportion))

    # init videowriter object (to put new frames into)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_name, fourcc, 30, dims)

    # get new sized frames
    for i in range(0, frame_count):
        # get orig video frame
        ret, frame = video.read()
        # resize it
        resized = cv2.resize(frame, dims, interpolation=cv2.INTER_AREA)
        # write to new video
        out.write(resized)

    out.release()
    video.release()

    print('Resized video!')


def to_grayscale(which_video):
    """
    given a video, outputs a the same video but resized (for input into double view)
    :param which_video: bool, 1 if day, 0 if night
    :return: Null
    """
    # import orig vid + give name to new vid TODO: fix path
    if which_video:
        video = cv2.VideoCapture('DOWNTOWN_DAY.mp4')
        out_name = 'DOWNTOWN_DAY_grayscale.avi'
    else:
        video = cv2.VideoCapture('DOWNTOWN_NIGHT.mp4')
        out_name = 'DOWNTOWN_NIGHT_grayscale.avi'

    # get important parameters
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # init videowriter object (to put new frames into)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_name, fourcc, 30, (int(w), int(h)))

    # get new sized frames
    for i in range(0, frame_count):
        # get orig video frame
        ret, frame = video.read()
        # resize it
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # write to new video
        out.write(grayed)

    out.release()
    video.release()




##### THIS ONE IS IN PROGRESS ####
# don't have the timing right yet
def write_fix_vid_double(subNum, which_video, time_range):
    """
    same as write_fix_vid, but with double view of entropy + regular vid at same time
    :param subNum: str, name of subject
    :param which_video: bool, 1 if day, 0 if night
    :param time_range: tuple, first is int of number of seconds into video to start, second is int of how many seconds total
    :return: Null
    """
    # get original video
    if which_video:
        ent_video = cv2.VideoCapture('DOWNTOWN DAY-entropy.mp4')
        reg_video = cv2.VideoCapture('DOWNTOWN_DAY_resized.avi')
        fix_file = subNum + '_day_fix.csv'
        out_name = subNum + '_day_fix_both.avi'
    else:
        ent_video = cv2.VideoCapture('DOWNTOWN NIGHT-entropy.mp4')
        reg_video = cv2.VideoCapture('DOWNTOWN_NIGHT_resized.avi')
        fix_file = subNum + '_night_fix.csv'
        out_name = subNum + '_night_fix_both.avi'

    # get width and height (should be same for both videos)
    w = ent_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h = ent_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    w_reg = reg_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h_reg = reg_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # check to make sure video frames are same size; else quit
    if w != w_reg or h != h_reg:
        print("Video dimensions don't match :(")
        return

    # import fixations --> orig fix.csv
    fix = genfromtxt(fix_file, delimiter=',')

    # init video writer obj
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (int(w), 2 * int(h))  # 2 x height since there's going to be two videos on top of each other
    out = cv2.VideoWriter(out_name, fourcc, 90, frame_size)

    # set color
    color = (0, 0, 255)

    # set start and end frame
    start_frame = time_range[0] * 90  # times 90 since 90 is sample rate TODO: change to variable later
    end_frame = (time_range[0]+time_range[1]) * 90

    # loop through while writing frames
    for i in range(start_frame, end_frame):  # first 1000 frames for testing # range(0, fix.shape[1]):
        # set some parameters
        isFix = fix[0, i]
        frame_num = int(fix[3, i])
        xcord = fix[4, i]
        ycord = fix[5, i]

        # get that specific frame for each video
        ent_video.set(1, frame_num)
        ent_ret, ent_frame = ent_video.read()
        reg_video.set(1, frame_num)
        reg_ret, reg_frame = reg_video.read()

        if isFix:  # if is fixation, draw fixation
            # get coordinates
            x = int(xcord * w)
            y = int(ycord * h)
            # draw cross hairs
            cv2.drawMarker(ent_frame, (x, y), color, markerType=0, thickness=5)
            cv2.drawMarker(reg_frame, (x, y), color, markerType=0, thickness=5)

        out_frame = cv2.vconcat([reg_frame, ent_frame])
        # write to outfile
        out.write(out_frame)

    ent_video.release()
    reg_video.release()
    out.release()

    print('Finished!')





######################################
"""
RUN TO GET REORG PICKLE AND ENTROPY VALUES FOR A SINGLE SUBJECT
"""
# conditions = ['day', 'night']
# subNum = 'pilot03'
# for each in conditions:
#     # set file name
#     file = subNum + '_' + each + '_fix.csv'
#
#     # get ordered fixations
#     fixations, timelapsed_reorg = reorg_fix(file)
#     fix_file_name = subNum + '_' + each + '_fix.p'
#     pickle.dump(fixations, open(fix_file_name, "wb"))
#     print('---- ' + each.upper() + ' ----')
#     print('Num of Fixations: ' + str(len(fixations)))
#     print('Reorg Time: ' + str(timelapsed_reorg))
#
#     # get entropy vals
#     entropy_vals, timelapsed_ave = get_entropy_values(1, fixations)
#     print('Ave Time: ' + str(timelapsed_ave))
#
#     # save to csv TODO: remove index
#     file_name = subNum + '_' + each + '_entropy_df.csv'
#     entropy_vals.to_csv(file_name)
#     print('Saved in csv')
#
#     # save as pickle
#     # file_name = subNum + '_' + each + '_entropy_df.p'
#     # pickle.dump(entropy_vals, open(file_name, "wb"))

# ---- DAY ----
# Num of Fixations: 7817
# Reorg Time: 0.6454579830169678
# Ave Time: 247.06952595710754
# ---- NIGHT ----
# Num of Fixations: 5642
# Reorg Time: 0.5103371143341064
# Ave Time: 176.39222812652588

######################################


######################################
"""
RUN TO GENERATE RESIZED VIDEOS
"""
# for each in range(0, 2):
#     resize_reg_vid(each, 0.5)

######################################



######################################
"""
RUN TO GENERATE DOUBLE VIDEOS --> in progress
"""
# subNum = 'pilot03'
# start = time.time()
# for each in range(0, 2):
#     if each == 1:  # day
#         time_range = (251, 25)  # start: 4:11, for 20 seconds
#     else:  # night
#         time_range = (244, 25)  # start: 4:04, for 20 seconds
#     write_fix_vid_double(subNum, each, time_range)
#     pause = time.time()
#     print('Finished 1 vid')
#     print('Time elapsed: ' + str(pause-start))
# print('All done!')
######################################



######################################
"""
RUN TO GET OVERALL AVERAGE ENTROPY FOR VIDEOS
"""
# conditions = ['night', 'day']
#
# for each, index in zip(conditions, range(0, 2)):  # index correlates to which_video in get_ave_entropy
#     average_entropy, timelapsed = get_ave_entropy(index)
#     print('Time lapsed: ' + str(timelapsed))
#
#     total_ave = sum(average_entropy) / len(average_entropy)
#     print('Overall average: ' + str(total_ave))
#
#     file_name = each + '_ave_entropy.p'
#     pickle.dump(average_entropy, open(file_name, "wb"))
#     print('dumped pickle')

    # file_name = each + '_ave_entropy.csv'
    # with open(file_name, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(average_entropy)
    # f.close()
    # print('wrote file')


# DAY Overall:
#     Time lapsed: 38893.776018857956
#     Overall average: 5.012844093587982
# NIGHT Overall:
#     Time lapsed: 28376.72372484207
#     Overall average: 4.547056894595313


# DAY 1000:
#     Time lapsed: 2724.0939729213715
#     Overall average: 4.643753442281445
# NIGHT 1000:
#     Time lapsed: 2691.322028875351
#     Overall average: 5.165260171280947


######################################
