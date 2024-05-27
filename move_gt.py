import os

files_list = []
# List all files with names starting with 'gt' recursively in the folder "Files"
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        for file in files:
            if file.startswith('gt'):
                # print(os.path.join(root, file))
                files_list.append(os.path.join(root, file))

list_files('Files')

os.makedirs('MOTChallengeEvalKit_cv_test/data/eval/divo/gt', exist_ok=True)

# Get all the video names
video_names = set()
for file in files_list:
    video_names.add((file.split('/')[1]).split('_')[0])

# Create the directories for the videos
for video_name in video_names:
    os.makedirs(f'MOTChallengeEvalKit_cv_test/data/eval/divo/gt/{video_name}', exist_ok=True)
    os.makedirs(f'MOTChallengeEvalKit_cv_test/data/eval/divo/gt/{video_name}/gt', exist_ok=True)


# Move the files to the respective directories
for file in files_list:
    video_name = (file.split('/')[1]).split('_')[0]
    viewname = (file.split('/')[1]).split('_')[1]
    os.rename(file, f'MOTChallengeEvalKit_cv_test/data/eval/divo/gt/{video_name}/gt/{viewname}.txt')