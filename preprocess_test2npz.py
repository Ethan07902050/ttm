import os, sys
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms
from tqdm import tqdm
###
# video_dir = "student_data/student_data/videos"
# audio_dir = "student_data/student_data/audios"
# test_seg_dir = "student_data/student_data/test/seg"
# test_bbox_dir = "student_data/student_data/test/bbox"
# test_data_dir = "test_data"

video_dir = sys.argv[1]
test_seg_dir = sys.argv[2]
test_bbox_dir = sys.argv[3]
output_root_dir = sys.argv[4]

audio_dir = os.path.join(output_root_dir, "dlcvchallenge1_audios")
test_data_dir = os.path.join(output_root_dir, "dlcvchallenge1_test_data")
###
if not os.path.isdir(test_data_dir):
    os.makedirs(test_data_dir)

class TTM_Inference_Dataset(Dataset):
    def __init__(self, video_dir, audio_dir, bbox_dir, seg_dir, image_H=96, image_W=9):
        
        template_list = list()
        for seg_csv_name in os.listdir(seg_dir):
            df = pd.read_csv(os.path.join(seg_dir, seg_csv_name))
            df['hashcode'] = seg_csv_name.split("_")[0]
            template_list.append(df)

        self.cases = pd.concat(template_list, axis=0, ignore_index=True)

        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.bbox_dir = bbox_dir

        self.image_size = (image_H, image_W) # H, W
        self.sample_rate = 16000

    def __len__(self):
        return len(self.cases)
    
    def crop_image(self, image, x1, y1, x2, y2):
        image = transforms.ToPILImage()(image)
        roi = transforms.functional.crop(
            image, 
            top = y1,
            left = x1,
            height = y2-y1,
            width = x2-x1,
        )
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return transform(roi)

    def __getitem__(self, index):
        item = self.cases.iloc[index]
        num_frames = item['end_frame'] - item['start_frame'] + 1
            
        # process video data
        bbox_df = pd.read_csv(os.path.join(self.bbox_dir, item['hashcode']+"_bbox.csv"))
        person_bbox = bbox_df.loc[
            (bbox_df['person_id']==item['person_id'])
            &(bbox_df['frame_id']>=item['start_frame'])
            &(bbox_df['frame_id']<=item['end_frame'])
        ]

        is_empty = np.zeros(num_frames, dtype=int)

        cap = cv2.VideoCapture(os.path.join(self.video_dir, item['hashcode']+".mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, item['start_frame'])

        image_list = list()
        for i, frame_id in enumerate(range(item['start_frame'], item['end_frame']+1)):
            success, image = cap.read() # image.shape: H, W, C
            bbox = person_bbox.loc[(person_bbox['frame_id']==frame_id)]
            if (int(bbox['x1'])==(-1) and int(bbox['y1'])==(-1) and int(bbox['x2'])==(-1) and int(bbox['y2'])==(-1)):
                is_empty[i] = 1
            else:
                image_list.append(self.crop_image(
                    image,
                    int(bbox['x1']), 
                    int(bbox['y1']), 
                    int(bbox['x2']), 
                    int(bbox['y2']),
                ))
        cap.release()
        if len(image_list) > 0:
            image = np.stack(image_list)
        else:
            image = np.array([])

        # process audio data
        crop_audio = torch.empty(0)
        if num_frames > 1:
            ori_audio, ori_sample_rate = torchaudio.load(os.path.join(self.audio_dir, item['hashcode']+'.wav'))
            audio_transform = torchaudio.transforms.Resample(ori_sample_rate, self.sample_rate)
            audio = audio_transform(ori_audio)
            
            onset, offset = int(item['start_frame'] / 30 * self.sample_rate), int((item['end_frame']) / 30 * self.sample_rate)
            crop_audio = audio[:, onset:offset]
            crop_audio = torch.mean(crop_audio, dim=0)

        return item['hashcode'], item['person_id'], item['start_frame'], item['end_frame'], image, crop_audio, is_empty, num_frames

dataset = TTM_Inference_Dataset(
    video_dir=video_dir,
    audio_dir=audio_dir,
    bbox_dir=test_bbox_dir,
    seg_dir=test_seg_dir,
    image_H=96,
    image_W=96,
)
for i in tqdm(range(len(dataset))):
    hashcode, person_id, start, end, image, audio, is_empty, num_frames = dataset[i]
    # print(hashcode, person_id, start, end, image.shape, audio.shape, is_empty.shape)
    # print(len(is_empty), image.shape[0] + is_empty.sum()) # should be the same (equals to frame num)
    
    np.savez(
        file=os.path.join(test_data_dir, str(hashcode)+"_"+str(person_id)+"_"+str(start)+"_"+str(end)),
        image=image.astype('float32'),
        audio=audio.numpy(),
        is_empty=is_empty
    )
