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
        #                                                        self.p_level, self.log_ratio, abs(self.log_ratio))

def read_items_tsv(file):
    items = []
    with open(file, "r") as f:
        # expect that first line gives totals
        total, total_1, total_2 = f.readline().strip().split("\t")
        total_1 = float(total_1)
        total_2 = float(total_2)
        for line in f.readlines():
            name, freq_1, freq_2 = line.strip().split("\t")
            items.append(Item(name, freq_1, freq_2))

    return items, total_1, total_2

def read_items_csv(file):
    terms = pd.read_csv(file)
    # first row contains totals
    total_1 = terms.iloc[0]["f_PR-BD"]
    total_2 = terms.iloc[0]["f_Reference"]
    print(total_1, total_2)
    terms.drop(0, inplace=True)
    print(terms)
    items = []
    for term, freq_1, freq_2 in zip(terms.term.to_list(), terms["f_PR-BD"].to_list(), terms["f_Reference"].to_list()):
        items.append(Item(term, freq_1, freq_2))

    return items, total_1, total_2

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

def test_item(item, total_1, total_2, expected_ll, expected_p, expected_p_corrected, expected_lr, expected_overused):
    item.compute_log_likelihood(total_1, total_2)
    item.compute_relative_frequency(total_1, total_2)
    item.compute_log_ratio(total_1, total_2)
    item.compute_significance(490364)

    print("log likelihood: expected: %.2f computed: %.2f" %(expected_ll, round(item.log_likelihood, 2)))
    print("p: %.4f (expected: %.4f), p(Bonferroni corrected): %.4f (expected: %.4f)" %
          (round(item.p, 4), expected_p, round(item.p_corrected, 4), expected_p_corrected))
    print("log ratio: expected: %.2f computed: %.2f" % (expected_lr, round(item.log_ratio, 2)))
    print("overused: expected: %s computed: %s" % (expected_overused, item.overused))

    assert round(item.log_likelihood, 2) == round(expected_ll, 2)
    assert round(item.p, 4) == round(expected_p, 4)
    assert round(item.p_corrected, 4) == round(expected_p_corrected, 4)
    assert round(item.log_ratio, 2) == round(expected_lr, 2)
    assert item.overused == expected_overused

def test():
    total_1 = float(17771448)
    total_2 = float(17479101)
    item_1 = Item("i", 1088469, 561184)
    test_item(item_1, total_1, total_2, 162899.39, 0.0, 0.0, 0.93, "+")
    item_2 = Item("factchecker", 0, 1)
    test_item(item_2, total_1, total_2, 1.40, 0.2362, 1.0, -1.02, "-")
    item_3 = Item("difficulties", 187, 92)
    test_item(item_3, total_1, total_2, 31.45, 0.0, 0.0101, 1.00, "+")

def run():
    file = sys.argv[1]

    items, total_1, total_2 = read_items_csv(file) # read_items_tsv(file)
    items = compute_log_likelihood_log_ratio(items, total_1, total_2, 94)
    write_statistics(items) #, p_level=99.99, log_ratio=1.0)

if __name__ == "__main__":
    # test()
    run()

