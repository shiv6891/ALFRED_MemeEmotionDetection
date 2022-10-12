import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

import os 
from tqdm import tqdm

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from PIL import Image

from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer, BertModel

from easydict import EasyDict

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


from transformers import ViTFeatureExtractor, ViTModel


# Set seed for reproducibility
torch.manual_seed(13)
np.random.seed(13)
random.seed(13)


train_path = './data/train_sample.csv'
val_path = 'ENTER VAL DATA PATH'
test_path = 'ENTER TEST DATA PATH'

image_dir = './data/sample_memes'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

le = LabelEncoder()

train_df['sent_target'] = le.fit_transform(train_df['sentiment'])
val_df['sent_target'] = le.transform(val_df['sentiment'])
test_df['sent_target'] = le.transform(test_df['sentiment'])


class MemeDataset(Dataset):
    
    def __init__(self, df, tokenizer, image_dir):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # Process the text
        text = self.df.iloc[idx]['ocr_text']

        text = ' '.join(text.split('\n'))
        
        encoded = self.tokenizer(text, padding='max_length', max_length=30, truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Process the image
        image_name = self.df.iloc[idx]['image']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        index, _ = os.path.splitext(image_name)

        image_inputs = self.feature_extractor(images=image, return_tensors='pt')
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        emo_ftrs_path = os.path.join('./data/sample_emotion_features', f"{index}.pt")
        emo_ftrs = torch.load(emo_ftrs_path).squeeze()

        
        
        # Process the labels
        target = torch.tensor(self.df.iloc[idx]['target'])
        

        sentiment = torch.tensor(self.df.iloc[idx]['sent_target'])

        return {
            'image': pixel_values, 
            'face': emo_ftrs,
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'target': target,
            'sentiment': sentiment
        }

from torch import Tensor
class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    """

    def __init__(self, alpha: float, n_classes: int, smoothing: float = 0.1):
        """
        :param alpha: Term for balancing soft_loss and hard_loss
        :param n_classes: Number of classes of the classification problem
        :param smoothing: Smoothing factor to be used during first epoch in soft_loss
        """
        super(OnlineLabelSmoothing, self).__init__()
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        # Initialize soft labels with normal LS for first epoch
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer('update', torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer('idx_count', torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor):
        # Calculate the final loss
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: Tensor, y: Tensor):
        """
        Calculates the soft loss and calls step
        to update `update`.
        :param y_h: Predicted logits.
        :param y: Ground truth labels.
        :return: Calculates the soft loss based on current supervise matrix.
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.
        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).
        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """
        This function should be called at the end of the epoch.
        It basically sets the `supervise` matrix to be the `update`
        and re-initializes to zero this last matrix and `idx_count`.
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()
        
class GatedCrossAttention(nn.Module):
    def __init__(self, args):
        super(GatedCrossAttention, self).__init__()
        self.args = args

        # linear for image-guided text attention
        self.img_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_1 = nn.Linear(args.hidden_dim, 1)

        # linear for text-guided image attention
        self.text_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.hidden_dim, 1)

    def forward(self, text_features, img_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        
        ############### 1. Image-guided text attention ###############
        # 1.1. Repeat the vectors -> [batch_size, num_img_region, max_seq_len, hidden_dim]
        text_features_rep = text_features.unsqueeze(1).repeat(1, self.args.num_img_region, 1, 1)
        img_features_rep = img_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, num_img_region, max_seq_len, hidden_dim]
        img_features_rep = self.img_linear_1(img_features_rep)

        
        # 1.3. sigmoid -> [batch_size, num_img_region, max_seq_len, hidden_dim]
        c_t = torch.sigmoid(img_features_rep)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, num_img_region, max_seq_len]
        alpha_t = self.att_linear_1(c_t).squeeze(-1)
        alpha_t = torch.softmax(alpha_t, dim=-1)
        

        # 1.5 Make new text vector with att matrix -> [batch_size, num_img_region, hidden_dim]
        f_t_hat = torch.matmul(alpha_t, text_features)  

        
        ############### 2. Text-guided visual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, num_img_region, num_img_region, hidden_dim]

        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.num_img_region, 1, 1)
        text_features_rep = f_t_hat.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)


        # 2.2 Feed to single layer (d*k) -> [batch_size, num_img_region, num_img_region, hidden_dim]
        text_features_rep = self.text_linear_2(text_features_rep)

        
        # 2.3. sigmoid -> [batch_size, num_img_region, num_img_region, hidden_dim]
        c_i = torch.sigmoid(text_features_rep)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, num_img_region, num_img_region]
        alpha_i = self.att_linear_2(c_i).squeeze(-1)
        alpha_i = torch.softmax(alpha_i, dim=-1)


        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        f_ei_hat = torch.matmul(alpha_i, img_features) 

        return f_t_hat, f_ei_hat
    

class LowRankBilinearPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.sum_pool = False
        self.proj1 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x1, x2):
        x1_ = self.nonlinearity(self.proj1(x1))
        x2_ = self.nonlinearity(self.proj2(x2))
        lrbp = self.proj(x1_ * x2_)
        return lrbp

class GMF(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF, self).__init__()
        self.args = args
        self.linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim) 
        self.linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.lrbp = LowRankBilinearPooling(args.hidden_dim)

    def forward(self, f_i, f_e):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        f_e = torch.tanh(self.linear_1(f_e))  # [b, m, 768]
        f_i = torch.tanh(self.linear_2(f_i))  # [b, m, 768]

        g_i = torch.sigmoid(self.lrbp(f_i, f_e))
        multimodal_features = torch.mul(g_i, f_e) + torch.mul(1 - g_i, f_i)  # [b, m, 768]

        return multimodal_features

class ALFRED(nn.Module):

    def __init__(self, num_classes: int, pretrained=True):
    
        super().__init__()
        

        self.num_classes = num_classes
        
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        

        self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        embed_dim = 768+768
        text_feature_dim = self.text_encoder.pooler.dense.in_features


        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc2 = nn.Linear((embed_dim//2), self.num_classes)
        
        self.leaky_relu = nn.LeakyReLU()
        
        args = EasyDict({
            'hidden_dim': 768,
            'max_seq_len': 30,
            'num_img_region': 197
        })

        self.gated_cross_attention = GatedCrossAttention(args)

        
        self.emo_gmf = GMF(args)


    def forward(self,  image, f_e, input_ids, attention_mask):

        batch_size = image.shape[0]

        output = self.text_encoder(input_ids, attention_mask, return_dict=True)

        img_output = self.visual_encoder(pixel_values=image)
        
        f_i = img_output.last_hidden_state
        
        f_ei = self.emo_gmf(f_i, f_e)
        
        
        f_t = output.last_hidden_state


        f_t_hat, f_ei_hat = self.gated_cross_attention(f_t, f_ei)

        z = torch.cat((f_t_hat, f_ei_hat), dim=2)
    
        joint_meme_repr = z.sum(1)


        x = self.leaky_relu(self.fc1(joint_meme_repr))
        
        y_hat= self.fc2(x)

        return y_hat


if __name__ == "__main__":

    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = MemeDataset(train_df, tokenizer, image_dir)
    val_dataset = MemeDataset(val_df, tokenizer, image_dir)
    test_dataset = MemeDataset(test_df, tokenizer, image_dir)

    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 16
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=32)
    
    model = ALFRED(6, pretrained=True)
    
    _ = model.cuda()
    
    params = list(model.parameters())
    
    criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=6, smoothing=0.1).cuda()
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    epochs = 20
    
    
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 16
    
    n_total_steps = len(train_loader)

    
    warnings.filterwarnings("ignore")
    for epoch in range(epochs):
        
        total_target = {
            'train': [],
            'test': [],
            'val': [],
        }
        
        total_preds = {
            'train': [],
            'test': [],
            'val': []
        }
        
        for i, batch in enumerate(tqdm(train_loader)):
            
            model.train()
            criterion.train()
            # Collect inputs
            image = batch['image'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['target'].cuda()
            face = batch['face'].cuda()
            
            logits = model(image, face, input_ids, attention_mask)
            
            loss = criterion(logits, labels.long())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(logits, 1)
            
            total_target['train'].extend(batch['target'].cpu().tolist())
            total_preds['train'].extend(preds.cpu().tolist())
            

                
        criterion.next_epoch()
                      
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            criterion.eval()

            model.eval()

            for j, batch in enumerate(tqdm(val_loader)):

                

                image = batch['image'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['target']
                face = batch['face'].cuda()

                logits = model(image, face, input_ids, attention_mask)


                _, preds = torch.max(logits, 1)

                total_target['val'].extend(batch['target'].cpu().tolist())
                total_preds['val'].extend(preds.cpu().tolist())
                
            for j, batch in enumerate(tqdm(test_loader)):


                image = batch['image'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['target']
                face = batch['face'].cuda()

                logits = model(image, face, input_ids, attention_mask)


                _, preds = torch.max(logits, 1)

                total_target['test'].extend(batch['target'].cpu().tolist())
                total_preds['test'].extend(preds.cpu().tolist())

            print()
            print("-"*40)
            print(f"Train Results Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            print(f"Macro F1: {f1_score(total_target['train'], total_preds['train'], average='weighted'):.4f}")
            print(f"Recall: {recall_score(total_target['train'], total_preds['train'], average='weighted'):.4f}")
            print(f"Precision: {precision_score(total_target['train'], total_preds['train'], average='weighted'):.4f}")
            print(f"Accuracy: {accuracy_score(total_target['train'], total_preds['train']):.4f}")
            print("-"*40)
            print()
            print("-"*40)
            print(f"Validation Results after Epoch [{epoch+1}/{epochs}]")
            print(f"Macro F1: {f1_score(total_target['val'], total_preds['val'], average='weighted'):.4f}")
            print(f"Recall: {recall_score(total_target['val'], total_preds['val'], average='weighted'):.4f}")
            print(f"Precision: {precision_score(total_target['val'], total_preds['val'], average='weighted'):.4f}")
            print(f"Accuracy: {accuracy_score(total_target['val'], total_preds['val']):.4f}")
            print("-"*40)
            print()
            print("-"*40)
            print(f"Test Results after Epoch [{epoch+1}/{epochs}]")
            print(f"Macro F1: {f1_score(total_target['test'], total_preds['test'], average='weighted'):.4f}")
            print(f"Recall: {recall_score(total_target['test'], total_preds['test'], average='weighted'):.4f}")
            print(f"Precision: {precision_score(total_target['test'], total_preds['test'], average='weighted'):.4f}")
            print(f"Accuracy: {accuracy_score(total_target['test'], total_preds['test']):.4f}")
            print("-"*40)
            print()