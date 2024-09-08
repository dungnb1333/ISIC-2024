import cv2
import albumentations as albu
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class ISIC_2018_Seg_Dataset(Dataset):
    def __init__(self, df, image_size, mode):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        assert mode in  ['train', 'valid']
        self.mode = mode
        if self.mode == 'train':
            self.df = df.sample(frac=1).reset_index(drop=True)
            self.transform = albu.Compose([
                albu.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.Affine(rotate=[-30, 30], shear=[-10, 10], interpolation=1, p=0.5),
                albu.Flip(p=0.5),
                albu.RandomRotate90(p=0.5),
                albu.OneOf([
                    albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
                    albu.GridDistortion(p=1),               
                ], p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.5),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.RandomBrightnessContrast(),
                ], p=0.5),
                albu.HueSaturationValue(p=0.5),
                albu.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.1),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].float().unsqueeze(0)
        mask /= 255.0
        
        return image, mask
        

class ISIC_2018_Seg_Test_Dataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform_256 = albu.Compose([
            albu.Resize(256, 256),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.transform_512 = albu.Compose([
            albu.Resize(512, 512),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])
        height, width = image.shape[0:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_256 = self.transform_256(image=image)['image']
        image_512 = self.transform_512(image=image)['image']
        return image_256, image_512, height, width, row['image_path']
        