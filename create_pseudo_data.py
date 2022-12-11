# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-09 02:07:41
#  * @modify date 2022-12-09 02:07:41
#  * @desc [description]
#  */


from build_model import DEVICE, IDX2TAG
from analysis import analysis_output

import torch
from torchmetrics.functional.classification import multiclass_accuracy


def gen_pseudo_data(
    model, domain_dev_iter, topn=300, save_output=False, output_file=None,
    csvsave=False, pngsave=False, csv_file_name=None, output_plot_name=None,
    device=DEVICE, num_classes=len(IDX2TAG), idx2tag=IDX2TAG):
  
  model.eval()

  new_words_lst = []
  new_tags_lst = []
  pseudo_tags_lst = []
  all_mean_prob_lst = []
  acc_lst = []

  with torch.no_grad():
      for i, batch in enumerate(domain_dev_iter):

        words, x, is_heads, tags, y, seqlens = batch

        # When calculating the length of sentences, ignore <pad>
        sen_len = y.bool().sum(axis=1)

        logits, _, y_hat = model(x, y)  # y_hat: (N, T)

        # Save prediction as new training dataset
        softmax_value = torch.softmax(logits, dim=2)
        max_prob = torch.amax(softmax_value, dim=2)

        # Rank by mean probability
        res_prob = y.bool().to(device) * max_prob.to(device)
        sum_prob = res_prob.sum(axis=1)
        mean_prob = sum_prob / sen_len.to(device)
        all_mean_prob_lst.extend(mean_prob.tolist())
        
        new_words = []
        new_tags = []
        pseudo_tags = []

        for words_i, tags_i, ishead_i, yhat_i in zip(words, tags, is_heads, y_hat):
          new_words.append(words_i.split()[1: -1])
          new_tags.append(tags_i.split()[1: -1])

          yhat_i = yhat_i.cpu().numpy().tolist()
          select_y_hat = [hat for head, hat in zip(ishead_i, yhat_i) if head == 1]
          preds = [idx2tag[hat] for hat in select_y_hat][1: -1]
          pseudo_tags.append(preds)

          assert len(preds)==len(words_i.split()[1: -1])==len(tags_i.split()[1: -1])

        new_words_lst.extend(new_words)
        new_tags_lst.extend(new_tags)
        pseudo_tags_lst.extend(pseudo_tags)

        # Calculate the accuracy for each sentences, ignore 0
        batch_acc = multiclass_accuracy(
            torch.tensor(y_hat).to(device), torch.tensor(y).to(device), num_classes=num_classes, 
            ignore_index=0, average="micro", multidim_average="samplewise")
        acc_lst.extend(batch_acc.tolist())

  ind = list(range(len(all_mean_prob_lst)))
  ind = [x for _, x in sorted(zip(all_mean_prob_lst, ind), reverse=True, key=lambda x: x[0])]
  prob_lst = [all_mean_prob_lst[i] for i in ind]

  select_ind = ind[: topn] # The index of topn sentences
  not_select_ind = ind[topn: ]

  top_words = [new_words_lst[i] for i in select_ind]
  top_tags = [new_tags_lst[i] for i in select_ind]
  top_pseudo_tags = [pseudo_tags_lst[i] for i in select_ind]

  # Save intermediate result - top n 
  if save_output:
    with open(output_file, 'w') as fout:
      for top_words_i, top_tags_i, top_pseudo_tags_i in zip(top_words, top_tags, top_pseudo_tags):
        for w, t, p in zip(top_words_i, top_tags_i, top_pseudo_tags_i):
          fout.write("{} {} {}\n".format(w, t, p))
        fout.write("\n")
  
  # Analysis the intermediate result
  _ = analysis_output(
      output_file, csvsave=csvsave, pngsave=pngsave, 
      csv_file_name=csv_file_name, output_plot_name=output_plot_name)

  remain_words = [new_words_lst[i] for i in not_select_ind]
  remain_tags = [new_tags_lst[i] for i in not_select_ind]
  remain_pseudo_tags = [pseudo_tags_lst[i] for i in not_select_ind]

  top_prob = prob_lst[: topn]
  remain_prob = prob_lst[topn: ]
  top_acc = [acc_lst[i] for i in select_ind]
  remain_acc = [acc_lst[i] for i in not_select_ind]

  return (top_words, top_tags, top_pseudo_tags, top_prob, top_acc,
          remain_words, remain_tags, remain_pseudo_tags, remain_prob, remain_acc)
  