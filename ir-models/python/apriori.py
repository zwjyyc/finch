from __future__ import division
import itertools


class Apriori:
    def __init__(self, min_support=0.3, min_confidence=0.1, n_item_rule=2):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.n_item_rule = n_item_rule
        self.assoc_rules = None
    # end constructor


    def _support(self, items, table):
        reduced_table = table
        reduced_users = table.index
        for item in items:
            reduced_table = reduced_table.loc[reduced_table.loc[:, item] > 0]
            reduced_users = reduced_table.index
        return len(reduced_users) / len(table.index)
    # end method


    def fit(self, table):
        reduced_items = []
        for item in list(table.columns):
            if self._support([item], table) > self.min_support:
                reduced_items.append(item)
        
        self.assoc_rules = []
        for rule in itertools.permutations(reduced_items, self.n_item_rule):
            from_item, to_item = rule[0], rule
            support_to_item = self._support(to_item, table)
            confidence = support_to_item / self._support([from_item], table)
            if confidence > self.min_confidence and support_to_item > self.min_support:
                self.assoc_rules.append(rule)
    # end method


    def predict(self):
        for items in self.assoc_rules:
            out = "who is using item %s is likely to use item" % items[0]
            for i in range(1, self.n_item_rule):
                out += ' %s' % items[i]
            print(out)
    # end method
