from prompt_rppg import fake_templates, real_templates
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(CrossAttentionLayer, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)

    def forward(self, input1, input2): # [1,128] [2,2]
        # Compute attention weights
        input1_embed = self.fc1(input1)
        input2_embed = self.fc2(input2)
               
        input2_embed = input2_embed.view(128, 2)
        # print(f"input1_embed = {input1_embed.shape}")# [1, 128]
        # print(f"input2_embed = {input2_embed.shape}")# [128, 2]
        
        attn_weights = torch.matmul(input1_embed, input2_embed) # [1, 128] [128, 2]
        attn_weights = F.softmax(attn_weights, dim=1)
        # print(f"attn_weights = {attn_weights.shape}") # [1, 2]

        # Apply attention weights to input2
        attended_input2 = torch.matmul(attn_weights, input2)# [1, 2]  [2, 2] = [1, 2]
        # print(f"input2 = {input2.shape}") # [2, 2])
        # print(f"attended_input2 = {attended_input2.shape}") # [1, 2]

        # Apply element-wise multiplication to combine inputs
        # print(f"input1 = {input1.shape}") # [1, 128])
        # print(f"attended_input2 = {attended_input2.shape}")
        input1 = input1.view(128, 1)
        combined = input1 * attended_input2 # [1, 128] [1, 2]\
        # print(f"combined = {combined.shape}")
        return combined


class PAD_Classifier(nn.Module):
    def __init__(self, PAE_net, downstream_net,model_text): # , target_net
        super(PAD_Classifier, self).__init__()
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.attention_layer = CrossAttentionLayer(input_dim1=128, input_dim2=2, hidden_dim=128)
        self.Extractor = PAE_net
        self.downstream = downstream_net
        self.classifier_rPPG_attention = nn.Linear(80, 2)
        self.classifier_rPPG_attention_stack = nn.Linear(122880, 160) 
        self.dropout = nn.Dropout(0.5)
        self.classifier_attended_feature_to_512 = nn.Linear(256, 512)
        self.cosine_similarity = nn.CosineSimilarity()
        self.classifier_rppg_cross_landmark_cat_prompt= nn.Linear(512+256+256, 2)# 768
        self.classifier_layer_attention = nn.Linear(256, 2)
        self.classifier_land = nn.Linear(64+64, 2)
        self.classifier_rPPG = nn.Linear(160, 2)
        self.text_encode = model_text
    def forward(self, video, landmark, landmark_diff,sim_or_dis,size):
        # plus noise in landmark,landmark_diff
        # 提取 (x, y) 点并对所有点进行随机位移的函数
        def extract_and_displace_all_points(tensor,shift_x,shift_y):
            displaced_tensor = tensor.clone()  # 创建张量的副本以进行修改
            for i in range(tensor.size(1)):  # 遍历行
                for j in range(0, tensor.size(2), 2):  # 以步长2遍历列
                    x = float(tensor[0, i, j])
                    y = float(tensor[0, i, j + 1])
                    # 对每个 (x, y) 点进行随机位移
                    x += random.uniform(shift_x,shift_y)
                    y += random.uniform(shift_x,shift_y)
                    # 将位移后的值放回新张量中
                    displaced_tensor[0, i, j] = x
                    displaced_tensor[0, i, j + 1] = y
            return displaced_tensor
        if size == 32: # c40
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
        elif size ==64: # c23
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
            
        # landmark 
        PAE_feature, PAE_feature_diff = self.Extractor(landmark, landmark_diff)
        # print(f"PAE_feature = {PAE_feature.shape}")  # [1,64]

        # rppg
        gra_sharp = 2.0
        rppg_x ,score1_x,score2_x,score3_x,feature_1,feature_2 = self.downstream(video,gra_sharp,size) #　rppg_x ,score1_x,score2_x,score3_x
        
        # text  tokenize 
        texts = clip.tokenize(fake_templates[random.randint(0, 5)]).cuda(non_blocking=True) #tokenize
        class_embeddings = self.text_encode.encode_text(texts) 
        
    
        # stack rppg feature
        downstream_features_detach = torch.cat([feature_1, feature_2],dim=1)
        # print(f"downstream_features_detach = {downstream_features_detach.shape}")#  torch.Size([1, 1280, 96])
        downstream_features_detach = downstream_features_detach.view(1,-1)
        # print(f"downstream_features_detach = {downstream_features_detach.shape}")#  torch.Size([1, 122880])
        downstream_features_detach = self.classifier_rPPG_attention_stack(downstream_features_detach)
        # print(f"downstream_features_detach = {downstream_features_detach.shape}")# [1,160]

        """
        feature_rppg_cross_landmark_cat_prompt
        """
        # landmark
        feature_landmark = torch.cat([PAE_feature, PAE_feature_diff],dim=1)
        # rppg 
        rppg_x_view = torch.cat([rppg_x, downstream_features_detach],dim=1)
        rppg_x_view = rppg_x.reshape(2,-1) #[1,320] -> [2,80]
        rppg_x_attention = self.classifier_rPPG_attention(rppg_x_view) # nn.Linear(80, 2)
        attended_feature_f = self.attention_layer(feature_landmark, rppg_x_attention)
        attended_feature_f = self.dropout(attended_feature_f)
        attended_feature_f = attended_feature_f.view(1, 256)

        """
        add  loss_cos_feature_text,loss_cos_feature_text_other which big
        """
        attended_feature_f_512 = self.classifier_attended_feature_to_512(attended_feature_f)
        loss_cos_feature_text = self.cosine_similarity(attended_feature_f_512,class_embeddings) # cls_
        loss_cos_feature_text_other = self.cosine_similarity(attended_feature_f_512,class_embeddings_other) 
        # print(f"loss_cos_feature_text={loss_cos_feature_text},loss_cos_feature_text_other={loss_cos_feature_text_other}")

        feature_rppg_cross_landmark_cat_prompt = torch.cat([attended_feature_f_512,class_embeddings.view(1, -1)],dim=1)
        # print(f"feature_rppg_cross_landmark_cat_prompt = {feature_rppg_cross_landmark_cat_prompt.shape}") #  torch.Size([1, 768])        
        
        out_rppg_cross_landmark_cat_prompt = self.classifier_rppg_cross_landmark_cat_prompt(feature_rppg_cross_landmark_cat_prompt) # rppg_landmar
        out_rppg_landmark = self.classifier_layer_attention(attended_feature_f)
        out_land = self.classifier_land(torch.cat([PAE_feature, PAE_feature_diff], dim=1))  
        out_rPPG = self.classifier_rPPG(rppg_x)
        # out_prompt
        return out_rppg_landmark, out_land, out_rPPG, rppg_x ,loss_cos_feature_text,loss_cos_feature_text_other ,out_rppg_cross_landmark_cat_prompt
