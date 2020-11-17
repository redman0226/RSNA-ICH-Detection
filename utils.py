def computeAUROC(dataGT, dataPRED, classCount):

    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC

def search_f1(output, target):
    max_result_f1_list = []
    max_threshold_list = []
    precision_list = []
    recall_list = []
    eps=1e-20
    target = target.type(torch.cuda.ByteTensor)

    # print(output.shape, target.shape)
    for i in range(output.shape[1]):

        output_class = output[:, i]
        target_class = target[:, i]
        max_result_f1 = 0
        max_threshold = 0

        optimal_precision = 0
        optimal_recall = 0

        for threshold in [x * 0.01 for x in range(0, 100)]:

            prob = output_class > threshold
            label = target_class > 0.5
            # print(prob, label)
            TP = (prob & label).sum().float()
            TN = ((~prob) & (~label)).sum().float()
            FP = (prob & (~label)).sum().float()
            FN = ((~prob) & label).sum().float()

            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            # print(precision, recall)
            result_f1 = 2 * precision  * recall / (precision + recall + eps)

            if result_f1.item() > max_result_f1:
                # print(max_result_f1, max_threshold)
                max_result_f1 = result_f1.item()
                max_threshold = threshold

                optimal_precision = precision
                optimal_recall = recall

        max_result_f1_list.append(round(max_result_f1,3))
        max_threshold_list.append(max_threshold)
        precision_list.append(round(optimal_precision.item(),3))
        recall_list.append(round(optimal_recall.item(),3))

    return max_threshold_list, max_result_f1_list, precision_list, recall_list