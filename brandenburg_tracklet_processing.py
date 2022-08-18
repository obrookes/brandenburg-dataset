import argparse
import glob
import os


def sort_data():
    for d in os.listdir(data_dir):
        if '.' not in d:
            conts = os.listdir(data_dir + '/' + d)
            curr_dir = data_dir + '/' + d
            organise_directory(conts, curr_dir)
            move_annotations_and_videos(conts, curr_dir, d)


def move_annotations_and_videos(conts, curr_dir, d):
    for item in conts:
        if '.json' in item:
            filepath = data_dir + '/' + d + '/' + item
            os.system('mv ' + filepath + ' ' + curr_dir + '/detections')

        video_exts = ['.mp4', '.MP4', '.avi', '.AVI']
        if any(ext in item for ext in video_exts):
            filepath = data_dir + '/' + d + '/' + item
            os.system('mv ' + filepath + ' ' + curr_dir + '/videos')


def organise_directory(conts, curr_dir):
    if 'detections' not in conts:
        os.mkdir(curr_dir + '/detections')
    if 'videos' not in conts:
        os.mkdir(curr_dir + '/videos')
    if 'tracks' not in conts:
        os.mkdir(curr_dir + '/tracks')
    if 'frames' not in conts:
        os.mkdir(curr_dir + '/frames')


def make_tracklets(make_vid):
    for d in os.listdir(data_dir):
        if '.' not in d:
            os.chdir(data_dir + '/' + d)
            print("Creating tracking information for " + d)
            os.system("python "
                      + root + "/brandenburg-tracking/track.py \
                        --detection_path=detections --video_path=videos --l_confidence=0.25 --h_confidence=0.85 \
                        --iou=0.45 --length=24 --outpath=tracks --animal_class=" + d.lower())

            if make_vid:
                for track in os.listdir(os.getcwd() + '/tracks/pkl'):
                    title = track.split('_track')[0]
                    vid = glob.glob(f"{os.getcwd() + '/videos/' + title}*")[0].split('videos/')[-1]
                    os.system("python "
                              + root + "/brandenburg-tracking/make_video.py \
                                --frame_path=frames \
                                --results=tracks/pkl/" + track + " \
                                --video=videos/" + vid)


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-make_video', type=bool, help='Generate a new video displaying the tracklet')
    parser.add_argument('--make_video', action='store_true')
    parser.set_defaults(make_video=False)

    return parser.parse_args()


def find_dataset():
    data_root = os.listdir(root + '/data')
    for d in data_root:
        if 'Brandenburg' in d:
            return root + "/data/" + d


if __name__ == '__main__':
    args = parse_args()
    root = os.getcwd()
    data_dir = find_dataset()
    sort_data()
    make_tracklets(args.make_video)
