import torch
def setMask(y,y_pred,proportions=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
    l = len(proportions)
    filtered_ys =[]
    filtered_pred_ys = []
    for i in range(0,l-1):
        if i<l-1:
            mask = torch.logical_and(y.ge(proportions[i]),y.lt(proportions[i+1]))
        else:
            mask = torch.logical_and(y.ge(proportions[i]), y.le(proportions[i+1]))
        filtered_y = torch.masked_select(y, mask)
        filtered_pred_y = torch.masked_select(y_pred, mask)
        filtered_ys.append(filtered_y)
        filtered_pred_ys.append(filtered_pred_y)
    return filtered_ys,filtered_pred_ys
def setMaskFLTADP(y_2_labels, pred_2,y_1, pred_1,future_day,a_feat_size,fine_grained,proportions=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    l = len(proportions)
    filtered_y_1s = []
    filtered_y_2s = []
    filtered_pred1s = []
    filtered_pred2s = []
    for i in range(0, l - 1):
        if i < l - 1:
            mask = torch.logical_and(y_2_labels.ge(proportions[i]), y_2_labels.lt(proportions[i + 1]))
        else:
            mask = torch.logical_and(y_2_labels.ge(proportions[i]), y_2_labels.le(proportions[i + 1]))
        filtered_y_2 = torch.masked_select(y_2_labels, mask)
        filtered_pred2 = torch.masked_select(pred_2, mask)
        # mask : [batch_size,future_day,a_feat_size]
        # y_1 : [batch_size,future_day,a_feat_size || 1]
        mask = mask.reshape((-1, 1, 1)).repeat(1, future_day,a_feat_size)
        filtered_y_1 = torch.masked_select(y_1, mask)
        filtered_pred1 = torch.masked_select(pred_1, mask)
        if fine_grained:
            filtered_y_1 = filtered_y_1.reshape(-1,future_day,a_feat_size)
            filtered_pred1 = filtered_pred1.reshape(-1,future_day,a_feat_size)
        else:
            filtered_y_1 = filtered_y_1.reshape(-1, future_day, 1)
            filtered_pred1 = filtered_pred1.reshape(-1,future_day,1)
        filtered_y_1s.append(filtered_y_1)
        filtered_y_2s.append(filtered_y_2)
        filtered_pred1s.append(filtered_pred1)
        filtered_pred2s.append(filtered_pred2)
    return filtered_y_1s, filtered_y_2s, filtered_pred1s, filtered_pred2s
if __name__ == '__main__':
    y=torch.rand(1,10)
    print(y)
    setMask(y,y)