# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-08 14:39:15
#  * @modify date 2022-12-08 14:39:15
#  * @desc [description]
#  */


from utils import TAG2IDX, IDX2TAG
from build_model import DEVICE

import torch
import numpy as np
import pandas as pd
from collections import Counter
from torchmetrics.functional.classification import multiclass_accuracy


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(0)

def save_sns_fig(each_class_df, output_plot_file):

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
    tag2idx=TAG2IDX, idx2tag=IDX2TAG, device=DEVICE):

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

  if csvsave:
    each_class_df.to_csv(csv_file_name, index=False)
  if pngsave:
    save_sns_fig(each_class_df, output_plot_name)

  return each_class_df