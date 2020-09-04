# given fixations, outputs ave entropy value for each fixation

import sys
# print('\n'.join(sys.path))
import numpy as np
import pickle
import pandas as pd
import os
import time
import csv
from numpy import genfromtxt
# for debugging, seaborn still isn't working....
try:
    import seaborn as sns  # TODO: install seaborn
except ModuleNotFoundError:
    print('seaborn why :(')
# have to add this to import cv2 and matplotlib and i have no idea why
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages')
try:
    import cv2  # import cv2xxx
except ModuleNotFoundError:
     print('cv2 epic fail :(')

try:
    from matplotlib import pyplot as plt
    from matplotlib import animation
except ModuleNotFoundError:
    print(':(')

sns.set()

"""
######## funcs in this file ########
get_ave_entropy: 
    - gets the average normalized entropy for each frame in video (for full video) (OR gets ave luminance)
get_min_vals:
    - gets min values for each frame in entropy videos
reorg_fix:
    - given a file name, outputs a list of list of dicts for input into get_entropy_values
get_one_entropy_val:
    - given an index for a point in the original fix file, returns the entropy for that moment of fixation
get_entropy_values:
    - gets ave entropy value for each fixation (for a subject)
write_fix_vid:
    - given fixations and condition, outputs a video of fixations layered on (entropy) video 
resize_reg_vid:
    - given a video, outputs the same video but resized (for input into double view)
to_grayscale:
    - given a video, outputs same video but grayscale (for luminance values)
write_fix_vid_double:
    - like write_fix_vid, but stacks the normal + entropy view on top of each other
get_frame_entropy_dist:
    - given a video and frame, returns a vector of every entropy value within that frame
write_animated_distribution:
    - generate .mp4 animation of changing distribution of full frame + where fixation is in that distribution
"""

# videos are stored elsewhere; this is the path to get to them
vid_path = '/Users/sydney/Documents/Research/SCCN/street_view_vids'
# and the path to get back here
return_path = '/Users/sydney/Documents/Research/SCCN/street_view'


def get_ave_entropy(which_video, which_color):
    """
    gets the average color value (entropy or luminance)
    test with: ave_entropy, timelapsed = get_ave_entropy(video_name)
    :param which_video: str, 'day' or 'night'
    :param which_color: str, 'entropy' or 'grayscale' (for luminance)
    :return ave_entropy, timelapsed: np array of ave entropy values (or luminance, i just didn't feel
                                     like changing the name) time taken to run
    """
    # check to see if inputs entered right
    if which_video != "day" and "night":
        print('Parameter for "which_video" entered incorrectly; please type "day" or "night"')
        quit()
    if which_color != "entropy" and "grayscale":
        print('Parameter for "which_color" entered incorrectly; please type "entropy" or "grayscale"')
        quit()

    start = time.time()

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_' + which_color + '.mp4'
    video = cv2.VideoCapture(vid_name)

    # return to og dir
    os.chdir(return_path)

    # get max vals
    max_vals = []
    if which_color == "entropy":
        max_vals_name = 'downtown_' + which_video + '_max_entropy.csv'
        max_vals = genfromtxt(max_vals_name, delimiter=',')

    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # init entropy/luminance array
    ave_entropy = np.zeros(frame_count)

    for i in range(0, frame_count):
        ret, frame = video.read()
        frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)
        big_val = 0
        for x in range(0, int(h_video)):
            for y in range(0, int(w_video)):
                pixel = frame[x, y]
                big_val = big_val + sum(pixel)

        # get average
        this_ave = (big_val / (w_video * h_video)) / 765

        # if entropy, multiply by max value
        if which_color == "entropy":
            this_ave = this_ave * max_vals[int(frame_num) - 1]

        # append to big main dataset
        ave_entropy[i] = this_ave

        if i % 1000 == 0:
            print('Completed ' + str(i) + ' frames')
            pause = time.time()
            print('Time elapsed: ' + str(pause - start))

    stop = time.time()
    timelapsed = stop - start

    return ave_entropy, timelapsed


def get_min_vals(which_video):
    start = time.time()

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_entropy.mp4'
    video = cv2.VideoCapture(vid_name)

    # return to og dir
    os.chdir(return_path)

    # get max vals
    max_vals_name = 'downtown_' + which_video + '_max_entropy.csv'
    max_vals = genfromtxt(max_vals_name, delimiter=',')

    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # init entropy array
    min_vals = np.zeros(frame_count)
    # func for calculating min
    get_min = lambda arr, j: sum(arr) / 255 / 3 * max_vals[j-1]  # where arr is array of pixel color vals, j is i

    for i in range(0, 300):  # frame_count):
        # get frame
        ret, frame = video.read()
        # get minimum val of first pixel
        min_val = get_min(frame[0, 0], i)
        # loop through every pixel in frame
        for x in range(0, int(h_video)):
            for y in range(0, int(w_video)):
                # check to see if that pixel val is less previously found minimum
                pix_min = get_min(frame[x, y], i)
                if pix_min < min_val:
                    min_val = pix_min  # if so, assign to min_val
        # once min val is found, append to big list
        min_vals[i] = min_val
        # print i once in awhile to see where we're at
        if i % 1000 == 0:
            print('Completed ' + str(i) + ' frames')
            pause = time.time()
            print('Time elapsed: ' + str(pause - start))

    stop = time.time()
    timelapsed = stop - start

    print('All done!')

    return min_vals, timelapsed


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

# doesn't work yet
def get_one_entropy_val(which_video, subNum, frame_num, x, y):

    os.chdir(return_path)
    # check to see if inputs entered right
    if (which_video != "day") and (which_video != 'night'):
        print('Parameter for "which_video" entered incorrectly; please type "day" or "night"')
        quit()

    frame_num = int(frame_num)

    # get fixations  # tODO: need this?
    fix_file = subNum + '_' + which_video + '_fix.csv'
    fix = genfromtxt(fix_file, delimiter=',')

    # get max vals
    max_vals_name = 'downtown_' + which_video + '_max_entropy.csv'
    max_vals = genfromtxt(max_vals_name, delimiter=',')

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_entropy.mp4'
    video = cv2.VideoCapture(vid_name)

    # get width and height of video
    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # set and read frame
    video.set(1, frame_num)
    ret, frame = video.read()

    # get x and y coords adjusted to frame
    x_coord = int(np.around((x * h_video), decimals=0))
    y_coord = int(np.around((1-y) * w_video, decimals=0))

    get_ent = lambda arr: sum(arr) / 255 / 3 * max_vals[frame_num]  # where arr is array of pixel color vals TODO: check this

    entropy_val = get_ent(frame[y_coord, x_coord])  # TODO: check this

    return entropy_val


def get_entropy_values(which_video, which_color, fixations):
    """
    gets ave entropy value for each fixation (OR luminance)
    test with
    :param which_video: str, "day" or "night" only
    :param which_color: str, "entropy" or "grayscale"
    :param fixations: list, generated from reorg_fix
    :return entropy, timelapsed: pandas df with important info, time taken to get it
    """
    # check to see if inputs entered right
    if which_video != "day" and "night":
        print('Parameter for "which_video" entered incorrectly; please type "day" or "night"')
        quit()
    if which_color != "entropy" and "grayscale":
        print('Parameter for "which_color" entered incorrectly; please type "entropy" or "grayscale"')
        quit()

    start = time.time()

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_entropy.mp4'
    video = cv2.VideoCapture(vid_name)

    # return to og dir
    os.chdir(return_path)

    # get max vals
    max_vals = []
    if which_color == "entropy":
        max_vals_name = 'downtown_' + which_video + '_max_entropy.csv'
        max_vals = genfromtxt(max_vals_name, delimiter=',')

    # get width and height of video
    w_video = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h_video = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # init pandas df for entropy vals
    ave_what = 'Ave_' + which_color
    cols = [ave_what, 'Start_Time', 'End_Time', 'Duration']  # TODO: add frame start and end?
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
            if which_color == "entropy":
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

    print('All done!')

    return entropy, timelapsed


def write_fix_vid(file, which_video):
    """
    given fixations and condition, outputs a video of fixations layered on (entropy) video
    :param file: str, file name of fixations  TODO: don't really need this, actually
    :param which_video: bool, 1 if day, 0 if night
    :return: Null
    """
    # change dir to get videos
    os.chdir(vid_path)

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
    # change dir to get videos
    os.chdir(vid_path)

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


# this doesn't work and i have no idea why
def to_grayscale(which_video):
    """
    given a video, outputs a the same video but resized (for input into double view)
    :param which_video: bool, 1 if day, 0 if night
    :return: Null
    """
    # change dir to get videos
    os.chdir(vid_path)

    # import orig vid + give name to new vid
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
    out = cv2.VideoWriter(out_name, fourcc, 30, (int(w), int(h)), isColor=False)

    for i in range(0, frame_count):
        # get orig video frame
        ret, frame = video.read()
        # convert to grayscale
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # write to new video
        out.write(grayed)

    out.release()
    video.release()


def write_fix_vid_double(subNum, which_video, time_range):
    """
    same as write_fix_vid, but with double view of entropy + regular vid at same time
    :param subNum: str, name of subject
    :param which_video: bool, 1 if day, 0 if night
    :param time_range: tuple, first is int of number of seconds into video to start, second is int of how many seconds total
    :return: Null
    """
    # change dir to get videos
    os.chdir(vid_path)

    # get original video
    if which_video:
        ent_video = cv2.VideoCapture('DOWNTOWN_DAY_entropy.mp4')
        reg_video = cv2.VideoCapture('DOWNTOWN_DAY_resized.avi')
        fix_file = subNum + '_day_fix.csv'
        out_name = subNum + '_day_fix_both.avi'
    else:
        ent_video = cv2.VideoCapture('DOWNTOWN_NIGHT_entropy.mp4')
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

    os.chdir(return_path)

    # import fixations --> orig fix.csv
    fix = genfromtxt(fix_file, delimiter=',')
    if which_video:
        max_vals = genfromtxt('downtown_day_max_entropy.csv', delimiter=',')
    else:
        max_vals = genfromtxt('downtown_night_max_entropy.csv', delimiter=',')

    os.chdir(vid_path)

    # init video writer obj
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (int(w), 2 * int(h))  # 2 x height since there's going to be two videos on top of each other
    out = cv2.VideoWriter(out_name, fourcc, 90, frame_size)

    # set color
    color = (0, 0, 255)

    # set start and end frame
    start_frame = time_range[0] * 90  # times 90 since 90 is sample rate TODO: change to variable later
    end_frame = (time_range[0] + time_range[1]) * 90

    last_ent_val = 0  # placeholder

    # loop through while writing frames
    for i in range(start_frame, end_frame):  # first 1000 frames for testing # range(0, fix.shape[1]):
        # set some parameters
        isFix = fix[0, i]
        frame_num = int(fix[3, i])
        xcord = fix[4, i]
        ycord = 1 - fix[5, i]

        # get that specific frame for each video
        ent_video.set(1, frame_num)
        ent_ret, ent_frame = ent_video.read()
        reg_video.set(1, frame_num)
        reg_ret, reg_frame = reg_video.read()

        # draw rectangle behind entropy
        cv2.rectangle(ent_frame, (40, 20), (170, 60), (255, 255, 255), -1)

        if isFix:  # if is fixation, draw fixation
            # get coordinates
            x = int(xcord * w)
            y = int(ycord * h)
            # get entropy value
            pixel = ent_frame[y, x]
            ent_val_norm = np.mean(pixel) / 256  # TODO: or, do sum, and later divide?
            ent_val = ent_val_norm * max_vals[int(frame_num)]  # multiply by max to get un-normed value
            # draw cross hairs
            cv2.drawMarker(ent_frame, (x, y), color, markerType=0, thickness=5)
            cv2.drawMarker(reg_frame, (x, y), color, markerType=0, thickness=5)
            # write text (entropy value)
            cv2.putText(ent_frame, str(np.around(ent_val, 4)), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0,), 1,
                        cv2.LINE_AA)
            # fill placeholder
            last_ent_val = ent_val
        else:  # else put value for last fixation
            # write text (entropy value)
            cv2.putText(ent_frame, str(np.around(last_ent_val, 4)), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0,),
                        1, cv2.LINE_AA)

        out_frame = cv2.vconcat([ent_frame, reg_frame])
        # write to outfile
        out.write(out_frame)

    ent_video.release()
    reg_video.release()
    out.release()

    print('Finished!')


def get_frame_entropy_dist(which_video, frame_num):
    """
    Note: one frame takes about 3.5 seconds
    :param which_video: str, either "day" or "night"
    :param frame_num: int, which frame number
    :return entropy: np array/vector about 600k long, of entropy values for every pixel in scene
    """
    os.chdir(return_path)

    # check to see if inputs entered right
    if (which_video != "day") and (which_video != "night"):
        print('Parameter for "which_video" entered incorrectly; please type "day" or "night"')
        quit()

    start = time.time()

    # get max values
    max_vals_name = 'downtown_' + which_video + '_max_entropy.csv'
    max_vals = genfromtxt(max_vals_name, delimiter=',')
    frame_max = max_vals[frame_num]  # will only be one val since its same frame

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_entropy.mp4'
    video = cv2.VideoCapture(vid_name)

    # get width and height
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    entropy = np.zeros(w * h)  # allocate space for every pixel in an np array

    # get frame
    video.set(1, frame_num)
    ret, frame = video.read()

    # func to get entropy
    get_ent = lambda arr: sum(arr) / 255 / 3 * frame_max  # where arr is array of pixel color vals % TODO: ok this is definitely wrong

    count = 0
    # loop through every pixel in frame
    for x in range(0, w):
        for y in range(0, h):
            # get entropy value for that pixel, put into entropy
            ent = get_ent(frame[y, x])  #TODO: check rest of script, since frame[column, row]
            entropy[count] = ent

            count += 1  # increase count

            # print count once in awhile to see where we're at
            # if count % 100000 == 0:
            #     print('Completed ' + str(count) + ' pixels')
            #     pause = time.time()
            #     print('Time elapsed: ' + str(pause - start))

    # print('All done!')

    return entropy

# doesn't work yet
def write_animated_distribution(which_video, subNum, start_time, how_long):
    """
    generate .mp4 animation of changing distribution of full frame + where fixation is in that distribution
    :param which_video: str, "day" or "night" only
    :param subNum: str, format of "pilot03"
    :param start_time: int, starting point of what part of video to look at, in seconds
    :param how_long: int, how long a portion of video to look at, in seconds
    :return: Null
    """
    # get video
    if (which_video != "day") and (which_video != "night"):
        print('Parameter for "which_video" entered incorrectly; please type "day" or "night"')
        quit()

    start = time.time()

    # set color of distribution
    if which_video == "day":
        dist_color = 'xkcd:red orange'
        fix_color = 'xkcd:bright blue'
    else:
        dist_color = 'xkcd:night blue'
        fix_color = 'xkcd:hot pink'

    # get fixations
    fix_file = subNum + '_' + which_video + '_fix.csv'
    fix = genfromtxt(fix_file, delimiter=',')
    # get max values
    max_vals_name = 'downtown_' + which_video + '_max_entropy.csv'
    max_vals = genfromtxt(max_vals_name, delimiter=',')

    # change dir to get videos
    os.chdir(vid_path)

    # get video
    vid_name = 'DOWNTOWN_' + which_video.upper() + '_entropy.mp4'
    video = cv2.VideoCapture(vid_name)

    # init figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # set consistent lims so frame size doesn't change
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, 0.6)

   # define interval
    start_index = int(start_time * 90)
    end_index = int(start_index + (how_long * 90))

    def animation_frame(index):
        plt.clf()
        # label axes/title
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Density')
        ax.set_title('{which} Entropy Distribution Frame by Frame'.format(which=which_video.capitalize()), size=20)
        # get frame number
        frame_num = int(fix[3, index])
        # get distribution array
        entropy_dist = get_frame_entropy_dist(which_video, frame_num)
        # plot
        sns.kdeplot(entropy_dist, shade=True, color=dist_color, label='Distribution of Entropy', legend=True)
        if fix[1, index]:
            # get entropy
            fix_ent = get_one_entropy_val(which_video, subNum, frame_num, fix[4, index], fix[5, index])
            # plot entropy
            point, = plt.plot(fix_ent, 0.015, color=fix_color, marker='*', markersize=15)
            point.set_label('Fixation')

        ax.legend(loc='upper left', facecolor='w')

    # class matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None,
    #               save_count=None, *, cache_frame_data=True, ** kwargs)[source]¶
    anim = animation.FuncAnimation(fig, func=animation_frame, frames=np.arange(start_index, end_index, 1))

    writermp4 = animation.FFMpegWriter(fps=30)
    anim_file_name = subNum + '_' + which_video + '_dist_animation.mp4'
    anim.save(anim_file_name, writer=writermp4)

    # start_frame = start_time * 30
    # end_frame = start_frame + (how_long * 30)
    #
    # # for each frame
    # for frame_num in range(start_frame, end_frame):
    #     # get distribution array
    #     entropy_dist = get_frame_entropy_dist(which_video, frame_num)
    #     # get indices of possible fixations
    #     check_these = np.where(fix[3, :] == frame_num)  # row 3 is Frame Nums
    #     for index in check_these:
    #         # plot distribution
    #         sns.kdeplot(entropy_dist, shade=True, color=dist_color, label=which_video, legend=True)
    #         # for each sample...
    #         if fix[1, index]:
    #             # get entropy
    #             fix_ent = get_one_entropy_val(which_video, subNum, frame_num, fix[4, index], fix[5, index])
    #             # plot entropy
    #             plt.plot(fix_ent, 0, 'ro')

    return



# for testing:
# ent = get_one_entropy_val('day', 'pilot03', 18, 0.4463802, 0.5167002)
