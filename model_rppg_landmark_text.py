from prompt_rppg import fake_templates, real_templates
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(CrossAttentionLayer, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)

    def forward(self, input1, input2): # [1,128] [2,2]

        input1_embed = self.fc1(input1)
        input2_embed = self.fc2(input2)
               
        input2_embed = input2_embed.view(128, 2)
        
        attn_weights = torch.matmul(input1_embed, input2_embed)
        attn_weights = F.softmax(attn_weights, dim=1)

        attended_input2 = torch.matmul(attn_weights, input2)

        input1 = input1.view(128, 1)
        combined = input1 * attended_input2

        return combined

class PAD_Classifier(nn.Module):
    def __init__(self, PAE_net, downstream_net,model_text):
        super(PAD_Classifier, self).__init__()
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.attention_layer = CrossAttentionLayer(input_dim1=128, input_dim2=2, hidden_dim=128)
        self.Extractor = PAE_net
        self.downstream = downstream_net
        self.classifier_rPPG_attention = nn.Linear(80, 2)
        self.classifier_rPPG_attention_stack = nn.Linear(122880, 160) 
        self.dropout = nn.Dropout(0.5)
        self.classifier_unified_feature_to_512 = nn.Linear(256, 512)
        self.cosine_similarity = nn.CosineSimilarity()
        self.classifier_rppg_cross_landmark_cat_prompt= nn.Linear(512+256+256, 2)# 768
        self.classifier_layer_attention = nn.Linear(256, 2)
        self.classifier_land = nn.Linear(64+64, 2)
        self.classifier_rPPG = nn.Linear(160, 2)
        self.text_encode = model_text
    
    def forward(self, video, landmark, landmark_diff,sim_or_dis,size):
        def extract_and_displace_all_points(tensor,shift_x,shift_y):
            displaced_tensor = tensor.clone() 
            for i in range(tensor.size(1)):  
                for j in range(0, tensor.size(2), 2): 
                    x = float(tensor[0, i, j])
                    y = float(tensor[0, i, j + 1])
                    x += random.uniform(shift_x,shift_y)
                    y += random.uniform(shift_x,shift_y)
                    displaced_tensor[0, i, j] = x
                    displaced_tensor[0, i, j + 1] = y
            return displaced_tensor
        if size == 32:
            landmark = extract_and_displace_all_points(landmark,-0.01,0.01)
            # lip 48 - 67
            for i in range(landmark.size(1)): 
                for j in range(98, 136, 2): 
                    x = float(landmark[0, i, j])
                    y = float(landmark[0, i, j + 1])
                    x += random.uniform(-0.006,+0.006)
                    y += random.uniform(-0.0085,+0.0085)
                    landmark[0, i, j] = x
                    landmark[0, i, j + 1] = y
            # eye 36 - 47
            for i in range(landmark.size(1)): 
                for j in range(74, 96, 2):  
                    x = float(landmark[0, i, j])
                    y = float(landmark[0, i, j + 1])
                    x += random.uniform(-0.0050,+0.0050)
                    y += random.uniform(-0.007,+0.007)
                    landmark[0, i, j] = x
                    landmark[0, i, j + 1] = y
            # Nose 27 - 35
            for i in range(landmark.size(1)):  
                for j in range(56, 72, 2): 
                    x = float(landmark[0, i, j])
                    y = float(landmark[0, i, j + 1])
                    x += random.uniform(-0.001,+0.001)
                    y += random.uniform(-0.0015,+0.0015)
                    landmark[0, i, j] = x
                    landmark[0, i, j + 1] = y
            landmark_diff = landmark_diff
        elif size == 64:
            landmark = extract_and_displace_all_points(landmark,-0.0005,0.0005)
            # lip 48 - 67
            for i in range(landmark.size(1)):  
                for j in range(98, 136, 2): 
                    x = float(landmark[0, i, j])
                    y = float(landmark[0, i, j + 1])
                    x += random.uniform(-0.0006,+0.0006)
                    y += random.uniform(-0.0005,+0.0005)
                    landmark[0, i, j] = x
                    landmark[0, i, j + 1] = y
            # eye 36 - 47
            for i in range(landmark.size(1)):  
                for j in range(74, 96, 2):  
                    x = float(landmark[0, i, j])
                    y = float(landmark[0, i, j + 1])
                    x += random.uniform(-0.0005,+0.0005)
                    y += random.uniform(-0.0004,+0.0004)
                    landmark[0, i, j] = x
                    landmark[0, i, j + 1] = y
            # Nose 27 - 35
            for i in range(landmark.size(1)): 
                for j in range(56, 72, 2): 
                    x = float(landmark[0, i, j])
                    y = float(landmark[0, i, j + 1])
                    x += random.uniform(-0.0001,+0.0001)
                    y += random.uniform(-0.0001,+0.0001)
                    landmark[0, i, j] = x
                    landmark[0, i, j + 1] = y
            landmark_diff = landmark_diff
        elif size ==128:
            pass
        else:
            print(f"Warning size")
            
        PAE_feature, PAE_feature_diff = self.Extractor(landmark, landmark_diff)

        # rppg
        gra_sharp = 2.0
        rppg_x ,score1_x,score2_x,score3_x,feature_1,feature_2 = self.downstream(video,gra_sharp,size) #　rppg_x ,score1_x,score2_x,score3_x
        
        # text tokenize 
        if sim_or_dis == 1:
            texts = clip.tokenize(real_templates[random.randint(0, 5)]).cuda(non_blocking=True) # tokenize
            other_texts = clip.tokenize(spoof_templates[random.randint(0, 5)]).cuda(non_blocking=True)
        elif sim_or_dis == -1:
            texts = clip.tokenize(spoof_templates[random.randint(0, 5)]).cuda(non_blocking=True) #tokenize
            other_texts = clip.tokenize(real_templates[random.randint(0, 5)]).cuda(non_blocking=True)
        else:
            print("Waring:self.sim_or_dis")

        # embed with text encoder
        class_embeddings = self.text_encode.encode_text(texts) 
        class_embeddings = class_embeddings.mean(dim=0) 
        class_embeddings_other = self.text_encode.encode_text(other_texts) 
        class_embeddings_other = class_embeddings_other.mean(dim=0) 

        # normalized features
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings_other = class_embeddings_other / class_embeddings_other.norm(dim=-1, keepdim=True)
        
        
        # stack rppg feature
        downstream_features_detach = torch.cat([feature_1, feature_2],dim=1)
        downstream_features_detach = downstream_features_detach.view(1,-1)
        downstream_features_detach = self.classifier_rPPG_attention_stack(downstream_features_detach)

        feature_landmark = torch.cat([PAE_feature, PAE_feature_diff],dim=1)
        rppg_x_view = torch.cat([rppg_x, downstream_features_detach],dim=1)
        rppg_x_view = rppg_x.reshape(2,-1)
        rppg_x_attention = self.classifier_rPPG_attention(rppg_x_view)
        unified_feature_f = self.attention_layer(feature_landmark, rppg_x_attention)
        unified_feature_f = self.dropout(unified_feature_f)
        unified_feature_f = unified_feature_f.view(1, 256)


        unified_feature_f_512 = self.classifier_unified_feature_to_512(unified_feature_f)
        loss_cos_feature_text = self.cosine_similarity(unified_feature_f_512, class_embeddings) 
        loss_cos_feature_text_other = self.cosine_similarity(unified_feature_f_512, class_embeddings_other) 
        feature_rppg_cross_landmark_cat_prompt = torch.cat([unified_feature_f_512, class_embeddings.view(1, -1)],dim=1)
        
        out_rppg_cross_landmark_cat_prompt = self.classifier_rppg_cross_landmark_cat_prompt(feature_rppg_cross_landmark_cat_prompt) 
        out_rppg_landmark = self.classifier_layer_attention(unified_feature_f)
        out_land = self.classifier_land(torch.cat([PAE_feature, PAE_feature_diff], dim=1))  
        out_rPPG = self.classifier_rPPG(rppg_x)

        return out_rppg_landmark, out_land, out_rPPG, rppg_x ,loss_cos_feature_text,loss_cos_feature_text_other ,out_rppg_cross_landmark_cat_prompt
