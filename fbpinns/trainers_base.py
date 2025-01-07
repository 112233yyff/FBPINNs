"""
Defines a generic base trainer class and extra training helper functions

This module is used by trainers.py
"""

import os
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
from tensorboardX import SummaryWriter

from fbpinns.util.logger import logger, switch_to_file_logger


class _Trainer:
    "Generic model trainer class"

    def __init__(self, c):
        "Initialise device and output directories"

        # clear directories
        c.get_outdirs()
        c.save_constants_file()
        logger.info(c)

        # initialise summary writer
        writer = SummaryWriter(c.summary_out_dir)
        writer.add_text("constants", str(c).replace("\n","  \n"))# uses markdown

        self.c, self.writer = c, writer
        # Initialize the file for saving i and loss
        self.training_log_path = self.c.summary_out_dir + "training_log.txt"
        with open(self.training_log_path, "w") as file:
            file.write("i,loss\n")  # Write the header

        # Initialize the file for saving i and time
        self.test_time_path = self.c.summary_out_dir + "test_time.txt"
        with open(self.test_time_path, "w") as file:
            file.write("i,time\n")  # Write the header

        # Initialize the file for saving i and lossval
        self.save_lossval_path = self.c.summary_out_dir + "save_lossval.txt"
        with open(self.save_lossval_path, "w") as file:
            file.write("i,lossval\n")  # Write the header


        # Initialize the file for saving i and array memory usage
        self.array_memory_usage_path = self.c.summary_out_dir + "array_memory_usage.txt"
        with open(self.array_memory_usage_path, "w") as file:
            file.write("i,array_memory_usage_KB\n")  # Write the header

    def array_save_memory(self, i, array_memory_usage):
        """
        保存内存使用情况到文件
        :param i: 当前训练步骤
        :param memory_usage: 当前内存使用量（单位：MB）
        """
        with open(self.array_memory_usage_path, "a") as file:
            file.write(f"{i},{array_memory_usage}\n")

    def _save_loss(self, i, l1):
        # Append i and loss to the file
        with open(self.training_log_path, "a") as file:
            file.write(f"{i},{l1}\n")

    def _save_lossval(self, i, lossval):
        # Append i and loss to the file
        with open(self.save_lossval_path, "a") as file:
            file.write(f"{i},{lossval}\n")

    def _save_time(self, i, time):
        # Append i and time to the file
        with open(self.test_time_path, "a") as file:
            file.write(f"{i},{time}\n")

    def _print_summary(self, i, loss, rate, start):
        "Prints training summary"

        logger.info("[i: %i/%i] loss: %.4f rate: %.1f elapsed: %.2f hr %s" % (
               i,
               self.c.n_steps,
               loss,
               rate,
               (time.time()-start)/(60*60),
               self.c.run,
                ))
        self.writer.add_scalar("loss/train", loss, i)
        self.writer.add_scalar("stats/rate", rate, i)

    def _save_figs(self, i, fs):
        "Saves figures"

        if self.c.clear_output: IPython.display.clear_output(wait=True)
        for name,f in fs:
            if self.c.save_figures:
                f.savefig(self.c.summary_out_dir+f"{name}_{i:08d}.png",
                          bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            self.writer.add_figure(name, f, i, close=False)
        if self.c.show_figures: plt.show()
        else: plt.close("all")

    def _save_model(self, i, model):
        "Saves a model"

        model = jax.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, model)# convert jax arrays to np
        with open(self.c.model_out_dir+f"model_{i:08d}.jax", "wb") as f:
            pickle.dump(model, f)

    def train(self):

        raise NotImplementedError


## HELPER FUNCTIONS

def train_models_multiprocess(ip, devices, c, Trainer, wait=0):
    "Helper function for training multiple runs at once (use with multiprocess.Pool)"

    # small hack so that tensorboard summaries appear in order
    time.sleep(wait)

    # switch logger to a file logger
    tag = os.environ["STY"].split(".")[-1] if "STY" in os.environ else "main"# grab socket name if using screen
    logfile = f"screenlog.{tag:s}.{ip:d}.log"
    switch_to_file_logger(logfile)

    # start training on specific device
    c.device = devices[ip]# set device to run on, based on process id
    c.show_figures = c.clear_output = False# make sure plots are not shown
    run = Trainer(c)
    run.train()




