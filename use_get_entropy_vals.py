
from get_entropy_values import get_ave_entropy
from get_entropy_values import get_entropy_values
from get_entropy_values import get_min_vals
from get_entropy_values import get_frame_entropy_dist
from get_entropy_values import get_one_entropy_val
from get_entropy_values import write_fix_vid_double
from get_entropy_values import write_animated_distribution
import os
from numpy import genfromtxt
import numpy as np
import pickle
import cv2
######################################

# ent = get_one_entropy_val('day', 'pilot03', 18, 0.4463802, 0.5167002)
# videos are stored elsewhere; this is the path to get to them
vid_path = '/Users/sydney/Documents/Research/SCCN/street_view_vids'
# and the path to get back here
return_path = '/Users/sydney/Documents/Research/SCCN/street_view'



# doesn't work yet
def get_one_lum_val(which_video, subNum, frame_num, x, y):

    os.chdir(return_path)
    # check to see if inputs entered right
    if (which_video != "day") and (which_video != 'night'):
        print('Parameter for "which_video" entered incorrectly; please type "day" or "night"')
        quit()

    frame_num = int(frame_num)

    # get fixations  # tODO: need this?
    fix_file = subNum + '_' + which_video + '_fix.csv'
    fix = genfromtxt(fix_file, delimiter=',')

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_grayscale.mp4'
    video = cv2.VideoCapture(vid_name)

    # get width and height of video
    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # set and read frame
    video.set(1, frame_num)
    ret, frame = video.read()

    # get x and y coords adjusted to frame
    height_coord = int(np.around((x * h_video), decimals=0))
    width_coord = int(np.around((1-y) * w_video, decimals=0))

    get_lum = lambda arr: sum(arr) / 255 / 3  # where arr is array of pixel color vals TODO: check this

    lum_val = get_lum(frame[height_coord, width_coord])  # TODO: check this

    return lum_val


# get fixations  # tODO: need this?
def get_fix_lum_vals(which_video, subNum):
    os.chdir(return_path)

    fix_file = subNum + '_' + which_video + '_fix.csv'
    fix = genfromtxt(fix_file, delimiter=',')

    entropies = np.zeros(5400)
    for i in range(0, 5400):
        if fix[0, i]:
            frame = int(fix[3, i])
            x = fix[4, i]
            y = fix[5, i]
            ent_val = get_one_lum_val(which_video, subNum, frame, x, y)
            entropies[i] = ent_val
        else:
            entropies[i] = np.NaN

    os.chdir(return_path)

    # save as pickle
    file_name = subNum + '_' + which_video + '_lum_fix.p'
    pickle.dump(entropies, open(file_name, "wb"))

    return entropies


p03_day_lum_fix = get_fix_lum_vals('day', 'pilot03')
p03_night_lum_fix = get_fix_lum_vals('night', 'pilot03')





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
RUN TO GENERATE GRAYSCALE VIDEOS
"""
# for each in range(0, 2):
#     to_grayscale(each)

######################################

######################################
"""
RUN TO GET MINIMUM VALUES
"""
# conditions = ['day', 'night']
# # for each in conditions:
# min_vals, timelapsed = get_min_vals('night')
# os.chdir(return_path)
# file_name = 'night_min_vals_300.p'
# pickle.dump(min_vals, open(file_name, "wb"))
# print('Time: ' + str(timelapsed/60))
# print('Finished')
######################################



######################################
"""
RUN TO GENERATE DOUBLE VIDEOS --> in progress
"""
# subNum = 'pilot03'
# start = time.time()
# for each in range(1, 2):
#     if each == 1:  # day
#         time_range = (251, 20)  # start: 4:11, for 20 seconds
#     else:  # night
#         time_range = (244, 20)  # start: 4:04, for 20 seconds
#     write_fix_vid_double(subNum, each, time_range)
#     pause = time.time()
#     print('Finished 1 vid')
#     print('Time elapsed: ' + str(pause-start))
# print('All done!')
######################################


######################################
"""
RUN TO GET OVERALL AVERAGE ENTROPY/LUMINANCE FOR VIDEOS
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
