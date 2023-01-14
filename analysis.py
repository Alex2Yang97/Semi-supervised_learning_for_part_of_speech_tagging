# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-08 14:39:15
#  * @modify date 2022-12-08 14:39:15
#  * @desc [description]
#  */

import os
from utils import TAG2IDX, IDX2TAG, RESULT_DIR
from build_model import DEVICE

import torch
import numpy as np
import pandas as pd
from collections import Counter
from torchmetrics.functional.classification import multiclass_accuracy


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

torch.manual_seed(0)

base_model_result = os.path.join(RESULT_DIR, "Base_model_result")

def save_sns_fig(each_class_df, output_plot_file, title=None):

  fig = plt.figure(figsize=(20,6))

  p1 = sns.scatterplot(
      data=each_class_df, x="POS_tags", y="acc",
      size = 8,
      legend=False)  

  for line in range(0, each_class_df.shape[0]):
      p1.text(
          each_class_df.POS_tags[line], each_class_df.acc[line], 
          round(each_class_df.acc[line], 3), horizontalalignment='left', 
          size='small', color='black', weight='semibold')

  sns.lineplot(data=each_class_df, x="POS_tags", y="acc")
  sns.scatterplot(data=each_class_df, x="POS_tags", y="acc")
  plt.xticks(rotation=45, size=10)
  if title:
    plt.title(title)

  ax2 = plt.twinx()
  sns.barplot(data=each_class_df, x="POS_tags", y="cnt", alpha=0.5, ax=ax2)

  fig.savefig(output_plot_file) 
  

def save_plotly_fig(each_class_df, output_plot_file):

  # Create figure with secondary y-axis
  fig = make_subplots(specs=[[{"secondary_y": True}]])

  # Add traces
  fig.add_trace(
      go.Bar(
          x=each_class_df["POS_tags"], y=each_class_df["cnt"], 
          name="count", opacity=0.5),
      secondary_y=True,
  )

  fig.add_trace(
      go.Scatter(
          x=each_class_df["POS_tags"], y=each_class_df["acc"], 
          mode='markers+lines', name="accuracy"),
      secondary_y=False,
  )

  # Add figure title
  fig.update_layout(
      title_text="Double Y Axis Example"
  )

  # Set x-axis title
  fig.update_xaxes(title_text="xaxis title")

  # Set y-axes titles
  fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
  fig.update_yaxes(title_text="<b>The number of tags</b>", secondary_y=True)
  fig.write_image(output_plot_file)


def analysis_output(
    output_res_file, csvsave=False, pngsave=False, csv_file_name=None, output_plot_name=None, 
    figtitle=None, tag2idx=TAG2IDX, idx2tag=IDX2TAG, device=DEVICE):

  y_true =  np.array([tag2idx[line.split()[1]] for line in open(output_res_file, 'r').read().splitlines() if len(line) > 0])
  y_pred =  np.array([tag2idx[line.split()[2]] for line in open(output_res_file, 'r').read().splitlines() if len(line) > 0])

  y_true_tensor = torch.from_numpy(y_true)
  y_pred_tensor = torch.from_numpy(y_pred)

  each_class_acc = multiclass_accuracy(
      torch.tensor(y_pred).to(device), 
      torch.tensor(y_true).to(device), 
      num_classes=len(tag2idx), 
      ignore_index=0, average=None)

  each_class_acc_lst = each_class_acc.tolist()

  each_class_df = pd.DataFrame.from_dict(Counter(y_true), orient='index').reset_index()
  each_class_df.columns = ["POS_id", "cnt"]
  each_class_df = each_class_df.sort_values(by="POS_id").reset_index(drop=True)
  each_class_df["acc"] = each_class_df["POS_id"].apply(lambda x: each_class_acc_lst[x])
  each_class_df["POS_tags"] = each_class_df["POS_id"].apply(lambda x: idx2tag[x])
  each_class_df = each_class_df[each_class_df["POS_tags"] != '<pad>'].reset_index(drop=True)

  tag_cnt_file = os.path.join(base_model_result, "tag_cnt_df.csv")
  tag_cnt_df = pd.read_csv(tag_cnt_file)[["POS_tags", "wsj_cnt"]]

  each_class_df = pd.merge(tag_cnt_df, each_class_df, on="POS_tags", how="left")
  each_class_df = each_class_df.fillna(0)

  if csvsave:
    each_class_df.to_csv(csv_file_name, index=False)
  if pngsave:
    save_sns_fig(each_class_df, output_plot_name, title=figtitle)

  return each_class_df


def plot_metric(precision, recall, f1, acc, metric_plot_path, title="", show=False, save=True):

  test_metric = pd.DataFrame({
      "Loop": list(range(len(precision))) * 4,
      "metric": ["precision"]*len(precision) + ["recall"]*len(recall) + ["f1"]*len(f1) + ["accuracy"]*len(acc),
      "value": precision + recall + f1 + acc
  })

  fig = px.line(test_metric, x="Loop", y="value", color='metric', markers=True, title=title)
  if save:
    fig.write_image(metric_plot_path, scale = 6, width = 1000, height = 500)
  if show:
    fig.show()
  return test_metric


def make_plot_metric(
    metrics_df, sub_metrics_dir, name, show=False, save=True):
  metric_plot_path = os.path.join(sub_metrics_dir, f"average-{name}.png")
  title = f"Metrics - average - {name}"
  _ = plot_metric(
      metrics_df["avg_domain_prec_lst"].tolist(), metrics_df["avg_domain_rec_lst"].tolist(), 
      metrics_df["avg_domain_f1_lst"].tolist(), metrics_df["avg_domain_acc_lst"].tolist(),
      metric_plot_path, title=name, show=show, save=save
      )
  
  metric_plot_path = os.path.join(sub_metrics_dir, f"micro-{name}.png")
  title = f"Metrics - micro - {name}"
  _ = plot_metric(
      metrics_df["micro_domain_prec_lst"].tolist(), metrics_df["micro_domain_rec_lst"].tolist(), 
      metrics_df["micro_domain_f1_lst"].tolist(), metrics_df["micro_domain_acc_lst"].tolist(),
      metric_plot_path, title=name, show=show, save=save
      )
  
  metric_plot_path = os.path.join(sub_metrics_dir, f"macro-{name}.png")
  title = f"Metrics - macro - {name}"
  _ = plot_metric(
      metrics_df["macro_domain_prec_lst"].tolist(), metrics_df["macro_domain_rec_lst"].tolist(), 
      metrics_df["macro_domain_f1_lst"].tolist(), metrics_df["macro_domain_acc_lst"].tolist(),
      metric_plot_path, title=name, show=show, save=save
      )