# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGAttN:
    @staticmethod
    def loss(logits, labels, nb_classes, class_weights):
        """
        logits: [batch_size, nb_classes]
        labels: [batch_size]
        class_weights: [nb_classes]
        """
        one_hot = F.one_hot(labels, nb_classes).float()
        sample_wts = torch.sum(one_hot * class_weights, dim=-1)
        xentropy = F.cross_entropy(logits, labels, reduction='none') * sample_wts
        return torch.mean(xentropy)

    @staticmethod
    def training(loss, lr, l2_coef, model):
        """
        返回优化器(PyTorch中不返回train_op,而是返回optimizer)
        """
        # Weight decay通过optimizer的weight_decay参数实现
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        return optimizer

    @staticmethod
    def preshape(logits, labels, nb_classes):
        """
        Reshape logits and labels
        """
        log_resh = logits.view(-1, nb_classes)
        lab_resh = labels.view(-1)
        return log_resh, lab_resh

    @staticmethod
    def confmat(logits, labels):
        """
        Confusion matrix
        """
        preds = torch.argmax(logits, dim=1)
        # PyTorch没有内置confusion_matrix,需要使用sklearn或自己实现
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())

    @staticmethod
    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = F.cross_entropy(logits, labels, reduction='none')
        mask = mask.float()
        mask = mask / mask.mean()
        loss = loss * mask
        return loss.mean()

    @staticmethod
    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Sigmoid cross-entropy loss with masking."""
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = loss.mean(dim=1)
        mask = mask.float()
        mask = mask / mask.mean()
        loss = loss * mask
        return loss.mean()

    @staticmethod
    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = (torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1))
        accuracy_all = correct_prediction.float()
        mask = mask.float()
        mask = mask / mask.mean()
        accuracy_all = accuracy_all * mask
        return accuracy_all.mean()

    @staticmethod
    def micro_f1(logits, labels, mask):
        """F1 score with masking."""
        predicted = torch.round(torch.sigmoid(logits))
        
        # Use integers
        predicted = predicted.long()
        labels = labels.long()
        mask = mask.long()
        
        # Expand mask for broadcasting
        mask = mask.unsqueeze(-1)
        
        # Count metrics
        tp = torch.sum(predicted * labels * mask)
        tn = torch.sum((predicted - 1) * (labels - 1) * mask)
        fp = torch.sum(predicted * (labels - 1) * mask)
        fn = torch.sum((predicted - 1) * labels * mask)
        
        # Calculate F1
        precision = tp.float() / (tp + fp).float()
        recall = tp.float() / (tp + fn).float()
        fmeasure = (2 * precision * recall) / (precision + recall)
        
        return fmeasure
