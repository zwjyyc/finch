from __future__ import division
import itertools


def support(items, table):
    reduced_table = table
    reduced_users = table.index
    for item in items:
        reduced_table = reduced_table.loc[reduced_table.loc[:, item] > 0]
        reduced_users = reduced_table.index
    return len(reduced_users) / len(table.index)


def apriori(table, min_support=0.3, min_confidence=0.1, n_item_rule=2):
    reduced_items = []
    for item in list(table.columns):
        if support([item], table) > min_support:
            reduced_items.append(item)
    
    assoc_rules = []
    for rule in itertools.permutations(reduced_items, n_item_rule):
        from_item, to_item = rule[0], rule
        support_to_item = support(to_item, table)
        confidence = support_to_item / support([from_item], table)
        if confidence > min_confidence and support_to_item > min_support:
            assoc_rules.append(rule)
    
    for items in assoc_rules:
        out = "who is watching %s is likely to watch" % items[0]
        for i in range(1, n_item_rule):
            out += ' %s' % items[i]
        print(out)
