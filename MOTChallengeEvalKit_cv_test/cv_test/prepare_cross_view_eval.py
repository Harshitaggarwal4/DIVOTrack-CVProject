import os
import numpy as np


gt_dir = "data/eval/divo/gt"
# track_dir = "data/eval/divo/mvmhat_result"
# save_dir = "data/eval/divo/mvmhat_result_cvma"
track_dir = "/media/bhavb/E Volume/Dev_Linux/CV/CV Project/Temperature/tracking_delta05_epsilon05" #track_with_conflict_free"
save_dir = "data/eval/divo/bhav"

gt_folder = "gt"

gt_box_type = 'xyxy'
delimiter = ' '

# gt_box_type = 'xywh'
# delimiter = ','

track_box_type = 'xyxy'
# track_box_type = 'fucked'
# track_box_type = 'xywh'
track_delimiter = ','

scale1 = 1
scale2 = 1

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

save_gt_dir = save_dir+"/gt"
if not os.path.exists(save_gt_dir):
	os.mkdir(save_gt_dir)

save_gt_cvma_dir = save_dir+"/gt_cvma"
if not os.path.exists(save_gt_cvma_dir):
	os.mkdir(save_gt_cvma_dir)

save_track_dir = save_dir+"/track"
if not os.path.exists(save_track_dir):
	os.mkdir(save_track_dir)

save_track_cvma_dir = save_dir+"/track_cvma"
if not os.path.exists(save_track_cvma_dir):
	os.mkdir(save_track_cvma_dir)

seq_map_path = save_dir+"/seqs.txt"
if os.path.exists(seq_map_path):
	os.remove(seq_map_path)
seq_map_file = open(seq_map_path, 'a')
seq_map_file.write('%s\n' % ('MOT16'))

scene_list = os.listdir(gt_dir)
for n in range(len(scene_list)):
	scene_id = scene_list[n]
	gt_vid_dir = gt_dir+"/"+scene_id+"/"+gt_folder
	track_vid_dir = track_dir+"/"+scene_id

	if 'Circle' in gt_vid_dir:
		continue

	seq_map_file.write('%s\n' % (scene_id))

	save_gt_scene_path = save_gt_dir+"/"+scene_id+".txt"
	if os.path.exists(save_gt_scene_path):
		os.remove(save_gt_scene_path)
	save_gt_file = open(save_gt_scene_path, 'a')


	save_gt_cvma_scene_path = save_gt_cvma_dir+"/"+scene_id+".txt"
	if os.path.exists(save_gt_cvma_scene_path):
		os.remove(save_gt_cvma_scene_path)
	save_gt_cvma_file = open(save_gt_cvma_scene_path, 'a')
		

	save_track_scene_path = save_track_dir+"/"+scene_id+".txt"
	if os.path.exists(save_track_scene_path):
		os.remove(save_track_scene_path)
	save_track_file = open(save_track_scene_path, 'a')


	save_track_cvma_scene_path = save_track_cvma_dir+"/"+scene_id+".txt"
	if os.path.exists(save_track_cvma_scene_path):
		os.remove(save_track_cvma_scene_path)
	save_track_cvma_file = open(save_track_cvma_scene_path, 'a')




	vid_list = os.listdir(gt_vid_dir)
	fr_cnt = 0
	for m in range(len(vid_list)):
		vid_name = vid_list[m]
		gt_path = gt_vid_dir+"/"+vid_name
		# print(vid_name, gt_path)

		# Bhav 28Apr
		# if "Drone" in vid_name:
		# 	vid_name = vid_name.replace("Drone", "View3")
		# if "View3" in vid_name:
		# 	vid_name = vid_name.replace("View3", "Drone")
  
		# new_names = {
		# 	"View1": "Drone",
		# 	"View2": "View1",
		# 	"View3": "View2",
		# }
		# new_names = {
		# 	"View1": "View1",
		# 	"View2": "View2",
		# 	"View3": "View3",
		# }
		new_names = {
			"Drone": "View1",
			"View1": "View2",
			"View2": "View3",
		}
		# new_names = {
		# 	"Drone": "Drone",
		# 	"View1": "View1",
		# 	"View2": "View2",
		# 	"View3": "View3",
		# }
		track_path = track_vid_dir+"/"+new_names[vid_name.split(".")[0]]+".txt"

		if "View2" in track_path:
			scale1=3640/1920
			scale2=2048/1080
		else:
			scale1=1
			scale2=1

		# convert format
		gt_data = np.loadtxt(gt_path, delimiter=delimiter, dtype=str)
		max_fr = 0
		for k in range(len(gt_data)):
			max_fr = max(max_fr, int(gt_data[k, 0]))
			# for self collect
			# if vid_list[m] == 'View1.txt':
			# 	gt_data[k, 2] = float(gt_data[k, 2]) * (1920 / 3640)
			# 	gt_data[k, 3] = float(gt_data[k, 3]) * (1080 / 2048)
			# 	gt_data[k, 4] = float(gt_data[k, 4]) * (1920 / 3640)
			# 	gt_data[k, 5] = float(gt_data[k, 5]) * (1080 / 2048)

			if gt_box_type=='xywh':
				save_gt_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(gt_data[k, 0])+fr_cnt, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
					float(gt_data[k, 4]), float(gt_data[k, 5]), -1, -1, -1, -1))

				save_gt_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(gt_data[k, 0])*len(vid_list)+m, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
					float(gt_data[k, 4]), float(gt_data[k, 5]), -1, -1, -1, -1))

			elif gt_box_type=='xyxy':
				save_gt_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(gt_data[k, 0])+fr_cnt, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
					float(gt_data[k, 4])-float(gt_data[k, 2]), float(gt_data[k, 5])-float(gt_data[k, 3]), -1, -1, -1, -1))

				save_gt_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(gt_data[k, 0])*len(vid_list)+m, int(gt_data[k, 1]), float(gt_data[k, 2]), float(gt_data[k, 3]),
					float(gt_data[k, 4])-float(gt_data[k, 2]), float(gt_data[k, 5])-float(gt_data[k, 3]), -1, -1, -1, -1))
		
		track_data = np.loadtxt(track_path, delimiter=track_delimiter, dtype=str)
		# print(track_data, track_path)

		for k in range(len(track_data)):
			max_fr = max(max_fr, int(track_data[k, 0]))

			if track_box_type=='xywh':
				save_track_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(track_data[k, 0])+fr_cnt, int(track_data[k, 1]), float(track_data[k, 2])*scale1, float(track_data[k, 3])*scale2,
					float(track_data[k, 4])*scale1, float(track_data[k, 5])*scale2, -1, -1, -1, -1))

				save_track_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(track_data[k, 0])*len(vid_list)+m, int(track_data[k, 1]), float(track_data[k, 2])*scale1, float(track_data[k, 3])*scale2,
					float(track_data[k, 4])*scale1, float(track_data[k, 5])*scale2, -1, -1, -1, -1))
			elif track_box_type=='xyxy':
				save_track_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(track_data[k, 0])+fr_cnt, int(track_data[k, 1]), float(track_data[k, 2])*scale1, float(track_data[k, 3])*scale2,
					(float(track_data[k, 4])-float(track_data[k, 2]))*scale1, (float(track_data[k, 5])-float(track_data[k, 3]))*scale2, -1, -1, -1, -1))

				save_track_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(track_data[k, 0])*len(vid_list)+m, int(track_data[k, 1]), float(track_data[k, 2])*scale1, float(track_data[k, 3])*scale2,
					(float(track_data[k, 4])-float(track_data[k, 2]))*scale1, (float(track_data[k, 5])-float(track_data[k, 3]))*scale2, -1, -1, -1, -1))
				
			elif track_box_type=='fucked':
				save_track_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(track_data[k, 0])+fr_cnt, int(track_data[k, 1]), ((float(track_data[k, 2]) + float(track_data[k, 4]))/2)*scale1, ((float(track_data[k, 3]) + float(track_data[k, 5]))/2)*scale2,
					abs((float(track_data[k, 2]) - float(track_data[k, 4])))*scale1, abs((float(track_data[k, 3]) - float(track_data[k, 5])))*scale2, -1, -1, -1, -1))

				save_track_cvma_file.write('%i, %i, %.2f, %.2f, %.2f, %.2f, %i, %i, %i, %i\n' 
					% (int(track_data[k, 0])*len(vid_list)+m, int(track_data[k, 1]), ((float(track_data[k, 2]) + float(track_data[k, 4]))/2)*scale1, ((float(track_data[k, 3]) + float(track_data[k, 5]))/2)*scale2,
					abs((float(track_data[k, 2]) - float(track_data[k, 4])))*scale1, abs((float(track_data[k, 3]) - float(track_data[k, 5])))*scale2, -1, -1, -1, -1))

		max_fr += 1
		fr_cnt += max_fr
	
	save_gt_file.close()
	save_track_file.close()
	save_gt_cvma_file.close()
	save_track_cvma_file.close()
 