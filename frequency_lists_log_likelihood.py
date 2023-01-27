"""
given a list of frequency counts for the same items from two sources (e.g. two corpora)
computes log likelihood and log ratio (effect size) as described here: http://ucrel.lancs.ac.uk/llwizard.html
"""
import sys
import math
import pandas as pd

import scipy.stats

zeroCorrectionLogRatio = 0.5
class Item(object):

    def __init__(self, name, freq_1, freq_2):
        self.name = name
        self.freq_1 = float(freq_1)
        self.freq_2 = float(freq_2)
        a = test

        self.log_likelihood = 0.0
        self.p = -0.0 # chi_sqare(self.log_likelihood)
        self.p_corrected = -0.0
        self.p_level = 0.0
        self.log_ratio = 0.0
        self.overused = "=" # indicate with + if it's overused in 1, otherwise "-"

    def compute_log_likelihood(self, total_1, total_2):
        """
        LL = 2* [D_1 + D_2]
        D_i = freq_i * ln(freq_i / E_i)
        E_i = total_i * E
        E = (freq_1 + freq_2) / (total_1 + total_2)
        :param total_1:
        :param total_2:
        :return:
        """
        # print("freq_1: %d freq_2: %d total_1: %d total_2: %d" %(self.freq_1, self.freq_2, total_1, total_2))

        E_1 = total_1 * (self.freq_1 + self.freq_2) / (total_1 + total_2)
        E_2 = total_2 * (self.freq_1 + self.freq_2) / (total_1 + total_2)
        # print("E1 %.2f, E2 %.2f" %(E_1, E_2))
        D_1 = self._compute_D_i(self.freq_1, E_1)
        D_2 = self._compute_D_i(self.freq_2, E_2)

        self.log_likelihood = 2* (D_1 + D_2)

    def compute_significance(self, num_comparisons):
        self.p = scipy.stats.chi2.sf(self.log_likelihood, 1)
        # Bonferroni correction
        self.p_corrected = min(num_comparisons * self.p, 1)
        if self.p_corrected < 0.001:
            self.p_level = 99.99
        elif self.p_corrected < 0.001:
            self.p_level = 99.9
        elif self.p_corrected < 0.01:
            self.p_level = 99.0
        elif self.p_corrected < 0.05:
            self.p_level = 95.0

    def _compute_D_i(self, freq, E_i):
        if freq == 0.0:
            return 0.0
        else:
            return freq * math.log(freq / E_i)

    def compute_relative_frequency(self, total_1, total_2):
        self.relative_freq_1 = self.freq_1 * 100 / total_1
        self.relative_freq_2 = self.freq_2 * 100 / total_2
        if self.relative_freq_1 > self.relative_freq_2:
            self.overused = "+"
        else:
            self.overused = "-"

    def _compute_norm_freq(self, freq, total):
        if freq == 0.0:
            freq = zeroCorrectionLogRatio
        norm_freq = freq / total
        return norm_freq

    def compute_log_ratio(self, total_1, total_2):
        """
        computes log ratio:
        logRatio = log2(N_1 / N2)
        N_i = f_i / total_i
        see: https://github.com/UCREL/SigEff/blob/master/C/sigeff.c
        :return:
        """
        self.log_ratio=math.log2(self._compute_norm_freq(self.freq_1, total_1) / self._compute_norm_freq(self.freq_2, total_2))

    def __str__(self, sep=";"):
        str_vals = [self.name, "%d" %self.freq_1, "%d" %self.freq_2, self.overused, "%.4f" %self.log_likelihood,
                    "%.4f" %self.p, "%.4f" %self.p_corrected, "%.2f" %self.p_level, "%.4f" %self.log_ratio,
                    "%.4f" %abs(self.log_ratio)]
        return sep.join(str_vals)
        #return "%s\t%d\t%d\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f" %(self.name, self.freq_1, self.freq_2, self.overused,
        #                                                        self.log_likelihood, self.p, self.p_corrected,
        #                                                        self.p_level, self.log_ratio, abs(self.log_ratio)



def compute_log_likelihood_log_ratio(items, total_1, total_2, num_comparisons):
    for item in items:
        item.compute_log_likelihood(total_1, total_2)
        item.compute_significance(num_comparisons)
        item.compute_relative_frequency(total_1, total_2)
        item.compute_log_ratio(total_1, total_2)
    return items # sorted(items, key=lambda i: i.log_likelihood, reverse=True)

def write_statistics(items, p_level=None, log_ratio = None):
    if p_level:
        items = [item for item in items if item.p_level >= p_level]
    if log_ratio:
        items = [item for item in items if abs(item.log_ratio) >= log_ratio]
    for item in items:
        print(item)

def run(items, total_1, total_2):
    items, total_1, total_2 = items, total_1, total_2
    items = compute_log_likelihood_log_ratio(items, total_1, total_2, 94)
    write_statistics(items) #, p_level=99.99, log_ratio=1.0)



