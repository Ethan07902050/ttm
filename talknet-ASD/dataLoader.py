import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms.functional as VF

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(audio, noiseAudio):   
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) == 0:
        return audio
    elif len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio
    return audio.astype(numpy.int16)

def load_audio(audio, numFrames, audioAug=False, noiseAudio=None):
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(audio, noiseAudio)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    fps = 30
    maxAudio = int(numFrames * 4)
    if len(audio) == 0:
        audio = np.zeros(maxAudio)
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)

    if audio.shape[0] < maxAudio:
        shortage = maxAudio - audio.shape[0]
        audio = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio

def load_visual(video, isEmpty, numFrames, visualAug=False): 
    H = 112
    # faces = np.zeros((numFrames, H, H))
    k = 0
    
    if len(video):
        video = np.einsum("jklm->jlmk", video)
    
    # for i in range(numFrames):
    #     if isEmpty[i]:
    #         faces[i] += 0.5 # images are normalized to mean 0.5, std 0.5
    #     else:
    #         frame = cv2.cvtColor(video[k], cv2.COLOR_BGR2GRAY)
    #         faces[i] = cv2.resize(frame, (H,H))
    #         k += 1
    # H = 112
    faces = []
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for i in range(numFrames):
        if isEmpty[i]:
            face = np.full((H, H), 0.5)
        else:
            face = cv2.cvtColor(video[k], cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (H,H))
            k += 1
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces

class train_loader(object):
    def __init__(self, files, batchSize, pathToFrameNum, aug=False):
        self.files  = files
        self.pathToFrameNum = pathToFrameNum
        self.miniBatch = []
        for file in files:
            data = np.load(file)
            self.pathToFrameNum[file] = len(data['is_empty'])
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(files, key=lambda file: self.pathToFrameNum[file], reverse=True)
        self.aug = aug
        start = 0        
        while True:
            length = self.pathToFrameNum[sortedMixLst[start]]
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList  = self.miniBatch[index]
        numFrames  = self.pathToFrameNum[batchList[-1]]
        audioFeatures, visualFeatures, labels = [], [], []
        if self.aug:
            fileToData = {}
            for file in batchList:
                data = np.load(file)
                fileToData[file] = data
            for file in batchList:
                video, audio, ttm, is_empty = fileToData[file]['image'], fileToData[file]['audio'], fileToData[file]['ttm'], fileToData[file]['is_empty']
                noiseName =  random.sample(set(list(fileToData.keys())) - {file}, 1)[0]
                noiseAudio = fileToData[noiseName]['audio']
                audioFeatures.append(load_audio(audio, numFrames, audioAug=True, noiseAudio=noiseAudio))
                visualFeatures.append(load_visual(video, is_empty,numFrames, visualAug=True))
                labels.append(ttm)
            return torch.FloatTensor(numpy.array(audioFeatures)), \
                torch.FloatTensor(numpy.array(visualFeatures)), \
                torch.LongTensor(numpy.array(labels))
        else:
            for file in batchList:
                data = np.load(file)
                video, audio, ttm, is_empty = data['image'], data['audio'], data['ttm'], data['is_empty']
                audioFeatures.append(load_audio(audio, numFrames, audioAug=False))
                visualFeatures.append(load_visual(video, is_empty,numFrames, visualAug=True))
                labels.append(ttm)
            return torch.FloatTensor(numpy.array(audioFeatures)), \
                torch.FloatTensor(numpy.array(visualFeatures)), \
                torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, files):
        self.files  = files
        self.miniBatch = [[file] for file in files] 

    def __getitem__(self, index):
        batchList  = self.miniBatch[index]
        audioFeatures, visualFeatures, labels = [], [], []
        for file in batchList:
            data = np.load(file)
            numFrames = len(data['is_empty'])
            video, audio, ttm, is_empty = data['image'], data['audio'], data['ttm'], data['is_empty']
            audioFeatures.append(load_audio(audio, numFrames, audioAug = False))  
            visualFeatures.append(load_visual(video, is_empty,numFrames, visualAug = False))
            labels.append(ttm)
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class test_loader(object):
    def __init__(self, files):
        self.files  = files
        self.miniBatch = [[file] for file in files] 

    def __getitem__(self, index):
        batchList  = self.miniBatch[index]
        audioFeatures, visualFeatures, ids = [], [], []
        for file in batchList:
            data = np.load(file)
            numFrames = len(data['is_empty'])
            video, audio, is_empty = data['image'], data['audio'], data['is_empty']
            audioFeatures.append(load_audio(audio, numFrames, audioAug = False))  
            visualFeatures.append(load_visual(video, is_empty,numFrames, visualAug = False))
            ids.append(os.path.basename(file)[:-4])
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               ids

    def __len__(self):
        return len(self.miniBatch)

class TTMDatasetWithBlank(Dataset):
    def __init__(self, files, mode="train"):
        self.files = files
        self.mode = mode # train / test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        image, audio, ttm, is_empty = data['image'], data['audio'], data['ttm'], data['is_empty']

        # process image frames
        full_image = np.zeros((len(is_empty), 3, 96, 96))
        k = 0
        for index, empty in enumerate(is_empty):
            if empty:
                full_image[index] += 0.5 # images are normalized to mean 0.5, std 0.5
            else:
                full_image[index] = image[k]
                k += 1
        return torch.from_numpy(full_image), torch.from_numpy(audio), torch.from_numpy(ttm)