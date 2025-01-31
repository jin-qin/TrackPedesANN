import cv2
import os
import log

class Visualizer:
    def __init__(self):
        var = 1
        # do something

    def visualizeProbabilityMap(self, map):
        #vis2 = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        im_color = cv2.applyColorMap(map, cv2.COLORMAP_BONE) #COLORMAP_BONE
        cv2.imshow('probability map colored', im_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def visualizeTestVideos(self, net, calLoader):

        log.log("Starting visualized testset videos..")

        video_sets = calLoader.get_video_file_names(False)

        for set_name, videos in video_sets.iteritems():

            log.log("Processing set {}".format(set_name))

            for video_name, video in videos.iteritems():

                frames = []
                ped_pos_init = []

                # each pedestrian has an unique id. each id/pedestrian will be processed only a single time.
                ped_keys = []

                log.log("Processing video {}".format(video_name))

                # read video frame by frame
                cap = cv2.VideoCapture(video)
                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frames.append(frame)

                annotations = calLoader.get_annotations()
                vid_frames = annotations[set_name][video_name]['frames']
                if not vid_frames is None:
                    for vid_i, vid_frame in vid_frames.iteritems():

                        frame_pos = []

                        for data in vid_frame:
                            ped_id = data['id']

                            if not ped_id in ped_keys:

                                # convert upper left corner to head position
                                data['pos'] = net.cornerToHead(data['pos'])

                                frame_pos.append(data['pos'])
                                ped_keys.append(ped_id)

                        ped_pos_init.append(frame_pos)

                # actual tracking + saving results in file
                log.log("Start live tracking: " + set_name + " " + video_name)
                net.live_tracking_video(frames, ped_pos_init, net.get_session_name() + "-" + set_name + "_" + video_name, 10)
                log.log("Finished live tracking: " + set_name + " " + video_name)